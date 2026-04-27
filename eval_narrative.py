"""
eval_length_ppl.py — Length-extrapolation probe on NarrativeQA.

Runs on the GPU cluster. No internet needed (consumes the .pt produced by
prepare.py and the checkpoint from train.py).

What it does:
    For each document, feeds the first `max_len` tokens through the model in
    a single forward pass with full causal attention, computes per-token NLL
    at every position, and saves:

        - raw NLLs:    nll_per_position.pt   (tensor [n_docs, max_len-1])
        - bucketed:    bucketed_ppl.csv      (avg NLL + PPL per position bucket)
        - plot:        length_vs_ppl.png     (PPL vs token position, log-y)

This is the standard length-extrapolation plot (cf. ALiBi, RoPE, YaRN papers):
a flat curve = good extrapolation, a hockey-stick = collapse.

Checkpoint loading mirrors train.py: torch.load -> TransformerConfig(**model_args)
-> GPT(cfg) -> load_state_dict (after stripping torch.compile's "_orig_mod." prefix).

By default, block_size in model_args is overridden to --max_len so the model can
accept longer sequences than it was trained on. This is the whole point. Pass
--keep_trained_block_size to disable.

Usage (run from your repo root, the dir that contains models/ and evals/):
    python eval_length_ppl.py \
        --ckpt out/<run_name>/checkpoint-120000/ckpt.pt \
        --tokens data/narrativeqa/narrativeqa_tokens.pt \
        --max_len 16384 \
        --out_dir results_mapformer/

Compare to baseline:
    python eval_length_ppl.py --ckpt out/.../ckpt.pt --tokens ... --out_dir results_baseline/
    python eval_length_ppl.py --ckpt out/.../ckpt.pt --tokens ... --out_dir results_mapformer/
    python plot_compare.py results_baseline results_mapformer
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from models.transformer_utils import TransformerConfig
from models.gpt import GPT
# ============================================================================
# MODEL LOADING — matches train.py checkpoint format
# ============================================================================
# Run this script from the repo root (the dir that contains models/, evals/, etc.)
# so these imports resolve. If you're running from elsewhere, set PYTHONPATH or
# pass --repo_root.
# def _import_model_classes(repo_root: str = None):
#     if repo_root is not None and repo_root not in sys.path:
#         sys.path.insert(0, repo_root)
#     from models.gpt import GPT
#     from models.transformer_utils import TransformerConfig
#     return GPT, TransformerConfig


def build_model(ckpt_path: str,
                device: torch.device,
                override_block_size: int = None,
                repo_root: str = None) -> torch.nn.Module:
    """
    Load a checkpoint produced by train.py.

    Checkpoint format (from train.py):
        {
          "model":          state_dict (may have "_orig_mod." prefix from torch.compile),
          "model_args":     dict passed to TransformerConfig,
          "iter_num":       int,
          "best_val_loss":  float,
          "config":         full training config dict,
          "optimizer":      ... (ignored here),
        }

    For length extrapolation, we OPTIONALLY override block_size in model_args so
    the model can accept sequences longer than what it was trained on. Whether
    this works without surgery depends on the architecture's positional code
    (RoPE / WM / EM / nWM / etc.). Pass override_block_size=None to keep the
    trained value.
    """
    # GPT, TransformerConfig = _import_model_classes(repo_root)

    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    print(f"[ckpt] iter_num={ckpt.get('iter_num', '?')} "
          f"best_val_loss={ckpt.get('best_val_loss', '?')}")
    print(f"[ckpt] model_args: {model_args}")

    trained_block_size = model_args.get("block_size", None)
    if override_block_size is not None and override_block_size != trained_block_size:
        print(f"[ckpt] OVERRIDE block_size: {trained_block_size} -> {override_block_size} "
              f"(length-extrapolation regime)")
        model_args["block_size"] = override_block_size

    cfg = TransformerConfig(**model_args)
    model = GPT(cfg)

    # Strip torch.compile prefix if present
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    fixed = 0
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            fixed += 1
    if fixed:
        print(f"[ckpt] stripped '_orig_mod.' prefix from {fixed} keys")

    # If we overrode block_size, the WPE / causal mask buffers in the new model
    # may have a different shape than the checkpoint's. load_state_dict with
    # strict=False so we don't choke; we'll print what's missing.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys (will use init values): {missing}")
    if unexpected:
        # Common case: ckpt has a wpe of trained block_size, model now has a
        # bigger one. We re-copy the trained slice into the new buffer below.
        print(f"[ckpt] unexpected keys (size mismatch?): {unexpected}")

    # Handle wpe (learned absolute position embedding) explicitly: copy the
    # trained slice into the front of the (possibly larger) new buffer. This is
    # only relevant for models that actually use it; for RoPE/WM/EM with no
    # learned absolute PE, there will be no "transformer.wpe.weight" and this
    # is a no-op.
    wpe_key = "transformer.wpe.weight"
    if wpe_key in ckpt["model"] and wpe_key in dict(model.named_parameters()):
        old_wpe = ckpt["model"][wpe_key]
        new_wpe = dict(model.named_parameters())[wpe_key]
        if old_wpe.shape != new_wpe.shape:
            with torch.no_grad():
                n = min(old_wpe.shape[0], new_wpe.shape[0])
                new_wpe[:n].copy_(old_wpe[:n])
            print(f"[ckpt] copied wpe[:{n}] from ckpt; positions >{n} are uninitialised")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ckpt] model loaded ({n_params:.1f}M params) on {device}")
    return model


@torch.no_grad()
def _forward_logits(model, input_ids):
    """
    nanoGPT-style models commonly take a fast path when targets=None and only
    return logits at the LAST position (shape [B, 1, V]). For per-token NLL we
    need logits at every position, so we pass dummy targets to force the model
    down the full-sequence branch. We then ignore the scalar loss it returns
    and compute per-position NLL ourselves.
    """
    out = model(input_ids, input_ids)   # dummy targets; we don't use the loss
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    # Sanity: if logits is still [B, 1, V], the model didn't take the bait.
    # Fall back to a clearer error rather than the cryptic batch-size mismatch.
    if logits.size(1) != input_ids.size(1):
        raise RuntimeError(
            f"Model returned logits of seq_len={logits.size(1)} for input seq_len="
            f"{input_ids.size(1)}. Your GPT.forward likely has a fast path that "
            f"only computes the last-position logits. Edit models/gpt.py to "
            f"return full-sequence logits, or modify _forward_logits in this script."
        )
    return logits


# ============================================================================
# EVAL
# ============================================================================
@torch.no_grad()
def nll_per_position(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Forward a single sequence and return NLL at every predicted position.

    input_ids: [1, T]
    returns:   [T-1]   (NLL of token t given tokens <t, for t=1..T-1)
    """
    logits = _forward_logits(model, input_ids)            # [1, T, V]

    shift_logits = logits[:, :-1, :].contiguous()         # [1, T-1, V]
    shift_targets = input_ids[:, 1:].contiguous()         # [1, T-1]

    # Per-token cross entropy in fp32 for stability
    nll = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction="none",
    )                                                     # [T-1]
    return nll.cpu()


def bucketize(nll: torch.Tensor, n_buckets: int = 32) -> list:
    """
    Geometric (log-spaced) buckets over position. Log-spaced is the right
    choice for length-extrapolation plots: most of the action is at the
    long-context tail, and linear bins waste resolution on short positions.

    Returns list of dicts: {start, end, mean_nll, ppl, n}.
    """
    T = nll.shape[-1]
    # bucket edges from 1..T, log-spaced
    edges = torch.unique(torch.logspace(0, math.log10(T), n_buckets + 1).long())
    edges[0] = 1
    edges[-1] = T

    out = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i].item()), int(edges[i + 1].item())
        if e <= s:
            continue
        chunk = nll[..., s:e]
        if chunk.numel() == 0:
            continue
        m = chunk.mean().item()
        out.append({
            "pos_start": s,
            "pos_end": e,
            "n_tokens": chunk.numel(),
            "mean_nll": m,
            "ppl": math.exp(m) if m < 50 else float("inf"),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (from train.py)")
    p.add_argument("--tokens", required=True, help="Path to narrativeqa_tokens.pt from prepare.py")
    p.add_argument("--max_len", type=int, default=16384, help="Max context length to probe")
    p.add_argument("--n_docs", type=int, default=None,
                   help="How many documents to evaluate (default: all that are long enough)")
    p.add_argument("--out_dir", default="results/")
    p.add_argument("--n_buckets", type=int, default=32)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--repo_root", default=None,
                   help="Path to the repo root (parent of models/, evals/). "
                        "Defaults to cwd; only set this if running the script from elsewhere.")
    p.add_argument("--keep_trained_block_size", action="store_true",
                   help="Don't override block_size to max_len. Use this if your model "
                        "uses a learned absolute position embedding and can't extrapolate.")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip matplotlib plotting (e.g. if not installed on cluster)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # ---- load model ----
    override_bs = None if args.keep_trained_block_size else args.max_len
    model = build_model(args.ckpt, device,
                        override_block_size=override_bs,
                        repo_root=args.repo_root)
    model.eval()

    # ---- load tokens ----
    print(f"[data] loading {args.tokens}")
    payload = torch.load(args.tokens, map_location="cpu", weights_only=False)
    docs = [d for d in payload["docs"] if d.numel() >= args.max_len]
    if not docs:
        raise RuntimeError(
            f"No documents in {args.tokens} reach max_len={args.max_len}. "
            f"Re-run prepare_data.py with --min_doc_tokens >= {args.max_len}."
        )
    if args.n_docs is not None:
        docs = docs[: args.n_docs]
    print(f"[data] using {len(docs)} documents (each truncated to {args.max_len} tokens)")

    # ---- run ----
    all_nlls = torch.empty(len(docs), args.max_len - 1, dtype=torch.float32)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype != torch.float32 else torch.cuda.amp.autocast(enabled=False)
    )

    for i, doc in enumerate(docs):
        ids = doc[: args.max_len].unsqueeze(0).to(device)   # [1, max_len]
        t0 = time.time()
        with autocast_ctx:
            nll = nll_per_position(model, ids)              # [max_len-1]
        all_nlls[i] = nll
        dt = time.time() - t0
        # quick sanity numbers: mean NLL in early window vs late window
        early = nll[:512].mean().item() if nll.numel() >= 512 else nll.mean().item()
        late = nll[-512:].mean().item() if nll.numel() >= 512 else nll.mean().item()
        print(f"[eval] doc {i+1}/{len(docs)}  {dt:5.1f}s  "
              f"early(<512) NLL={early:.3f} ppl={math.exp(min(early,50)):.1f}  "
              f"late(>{args.max_len-512}) NLL={late:.3f} ppl={math.exp(min(late,50)):.1f}")

    # ---- save raw ----
    raw_path = os.path.join(args.out_dir, "nll_per_position.pt")
    torch.save({
        "nll": all_nlls,                      # [n_docs, max_len-1]
        "max_len": args.max_len,
        "ckpt": args.ckpt,
        "tokens_file": args.tokens,
    }, raw_path)
    print(f"[save] raw NLLs -> {raw_path}  shape={tuple(all_nlls.shape)}")

    # ---- bucketed summary ----
    mean_nll_per_pos = all_nlls.mean(dim=0)   # [max_len-1], averaged over docs
    buckets = bucketize(mean_nll_per_pos, n_buckets=args.n_buckets)

    csv_path = os.path.join(args.out_dir, "bucketed_ppl.csv")
    with open(csv_path, "w") as f:
        f.write("pos_start,pos_end,n_tokens,mean_nll,ppl\n")
        for b in buckets:
            f.write(f"{b['pos_start']},{b['pos_end']},{b['n_tokens']},"
                    f"{b['mean_nll']:.6f},{b['ppl']:.6f}\n")
    print(f"[save] bucketed -> {csv_path}")

    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "max_len": args.max_len,
            "n_docs": len(docs),
            "global_mean_nll": float(all_nlls.mean()),
            "global_ppl": math.exp(min(float(all_nlls.mean()), 50)),
            "buckets": buckets,
        }, f, indent=2)
    print(f"[save] summary -> {json_path}")

    # ---- plot ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [(b["pos_start"] + b["pos_end"]) / 2 for b in buckets]
            ys = [b["ppl"] for b in buckets]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(xs, ys, marker="o")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Token position (context length)")
            ax.set_ylabel("Perplexity")
            ax.set_title(f"Length vs. PPL — {os.path.basename(args.ckpt)}\n"
                         f"{len(docs)} NarrativeQA docs, max_len={args.max_len}")
            ax.grid(True, which="both", ls="--", alpha=0.4)
            fig.tight_layout()
            png_path = os.path.join(args.out_dir, "length_vs_ppl.png")
            fig.savefig(png_path, dpi=150)
            print(f"[save] plot -> {png_path}")
        except ImportError:
            print("[plot] matplotlib not available, skipping (use --no_plot to silence)")


if __name__ == "__main__":
    main()