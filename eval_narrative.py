"""
eval_length_ppl.py — Length-extrapolation probe with INDEPENDENT CHUNKS.

Key difference from standard eval:
    Each chunk is processed INDEPENDENTLY - no shared context between chunks.
    This tests whether the model can handle various context lengths without
    positional encoding issues (e.g., RoPE wraparound at long contexts).

What it does:
    For each document, splits it into independent chunks of varying sizes
    (e.g., [0:512], [512:1024], [1024:2048], ..., [8192:16384])
    and computes PPL on each chunk SEPARATELY.

    Saves:
        - raw NLLs per chunk:  nll_per_chunk.pt
        - bucketed by length:  bucketed_ppl.csv  (PPL vs chunk size)
        - plot:                length_vs_ppl.png (PPL vs context length)
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


def build_model(ckpt_path: str,
                device: torch.device,
                override_block_size: int = None,
                repo_root: str = None) -> torch.nn.Module:
    """
    Load a checkpoint produced by train.py.
    [Same as before - keeping for completeness]
    """
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

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys (will use init values): {missing}")
    if unexpected:
        print(f"[ckpt] unexpected keys (size mismatch?): {unexpected}")

    # Handle wpe
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
    """Get full-sequence logits."""
    out = model(input_ids, input_ids)
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    if logits.size(1) != input_ids.size(1):
        raise RuntimeError(
            f"Model returned logits of seq_len={logits.size(1)} for input seq_len="
            f"{input_ids.size(1)}. Your GPT.forward likely has a fast path that "
            f"only computes the last-position logits."
        )
    return logits


@torch.no_grad()
def nll_for_chunk(model, chunk_ids: torch.Tensor) -> float:
    """
    Compute mean NLL for a SINGLE INDEPENDENT CHUNK.
    
    chunk_ids: [1, chunk_len]
    returns:   scalar mean NLL for this chunk
    """
    logits = _forward_logits(model, chunk_ids)        # [1, chunk_len, V]
    
    shift_logits = logits[:, :-1, :].contiguous()     # [1, chunk_len-1, V]
    shift_targets = chunk_ids[:, 1:].contiguous()     # [1, chunk_len-1]
    
    # Mean cross entropy for this chunk
    nll = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction="mean",  # Mean over this chunk
    )
    return nll.item()


def generate_chunk_ranges(max_len: int, min_chunk_size: int = 512) -> list:
    """
    Generate log-spaced chunk ranges for independent evaluation.
    
    Returns list of (start, end) tuples defining non-overlapping chunks.
    
    Example for max_len=8192, min_chunk_size=512:
        [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]
    
    This gives you PPL measurements at context lengths: 512, 512, 1024, 2048, 4096
    """
    chunks = []
    current_pos = 0
    chunk_size = min_chunk_size
    
    while current_pos < max_len:
        end_pos = min(current_pos + chunk_size, max_len)
        if end_pos - current_pos >= min_chunk_size:  # Only add if chunk is big enough
            chunks.append((current_pos, end_pos))
        current_pos = end_pos
        chunk_size *= 2  # Exponentially growing chunk sizes
    
    return chunks


def bucketize_by_length(chunk_results: list, n_buckets: int = 32) -> list:
    """
    Bucket chunks by their LENGTH (not position) and average NLL within buckets.
    
    chunk_results: list of dicts with keys {start, end, length, nll}
    returns: list of dicts with keys {length_min, length_max, mean_nll, ppl, n_chunks}
    """
    if not chunk_results:
        return []
    
    # Sort by chunk length
    sorted_chunks = sorted(chunk_results, key=lambda x: x["length"])
    min_len = sorted_chunks[0]["length"]
    max_len = sorted_chunks[-1]["length"]
    
    # Log-spaced bucket edges
    if max_len == min_len:
        edges = [min_len, max_len]
    else:
        edges = torch.unique(
            torch.logspace(math.log10(min_len), math.log10(max_len), n_buckets + 1).long()
        ).tolist()
    
    buckets = []
    for i in range(len(edges) - 1):
        len_min, len_max = edges[i], edges[i + 1]
        
        # Find all chunks in this length range
        chunks_in_bucket = [
            c for c in sorted_chunks 
            if len_min <= c["length"] < len_max or (i == len(edges) - 2 and c["length"] == len_max)
        ]
        
        if not chunks_in_bucket:
            continue
        
        mean_nll = sum(c["nll"] for c in chunks_in_bucket) / len(chunks_in_bucket)
        buckets.append({
            "length_min": len_min,
            "length_max": len_max,
            "n_chunks": len(chunks_in_bucket),
            "mean_nll": mean_nll,
            "ppl": math.exp(min(mean_nll, 50)),
            "std_nll": torch.tensor([c["nll"] for c in chunks_in_bucket]).std().item() if len(chunks_in_bucket) > 1 else 0.0,
        })
    
    return buckets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (from train.py)")
    p.add_argument("--tokens", required=True, help="Path to narrativeqa_tokens.pt from prepare.py")
    p.add_argument("--max_len", type=int, default=16384, help="Max document length to evaluate")
    p.add_argument("--min_chunk_size", type=int, default=512, 
                   help="Minimum chunk size (smaller chunks are too noisy)")
    p.add_argument("--n_docs", type=int, default=None,
                   help="How many documents to evaluate (default: all that are long enough)")
    p.add_argument("--out_dir", default="results/")
    p.add_argument("--n_buckets", type=int, default=20)
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep_trained_block_size", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    # ---- load model ----
    override_bs = None if args.keep_trained_block_size else args.max_len
    model = build_model(args.ckpt, device, override_block_size=override_bs)
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
    print(f"[data] using {len(docs)} documents (each up to {args.max_len} tokens)")

    # ---- generate chunk ranges ----
    chunk_ranges = generate_chunk_ranges(args.max_len, args.min_chunk_size)
    print(f"[eval] will evaluate {len(chunk_ranges)} independent chunks per document:")
    for start, end in chunk_ranges:
        print(f"       [{start:>5}:{end:>5}] (length={end-start})")

    # ---- run evaluation ----
    all_chunk_results = []  # Will store all chunks from all docs
    
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype != torch.float32 else torch.cuda.amp.autocast(enabled=False)
    )

    for doc_idx, doc in enumerate(docs):
        doc = doc[:args.max_len]  # Truncate to max_len
        print(f"\n[eval] Document {doc_idx + 1}/{len(docs)}")
        
        for chunk_idx, (start, end) in enumerate(chunk_ranges):
            if end > len(doc):
                print(f"  chunk {chunk_idx + 1}/{len(chunk_ranges)} [{start}:{end}] "
                      f"SKIPPED (doc too short)")
                continue
            
            chunk = doc[start:end].unsqueeze(0).to(device)  # [1, chunk_len]
            chunk_len = end - start
            
            t0 = time.time()
            with autocast_ctx:
                nll = nll_for_chunk(model, chunk)
            dt = time.time() - t0
            
            ppl = math.exp(min(nll, 50))
            print(f"  chunk {chunk_idx + 1}/{len(chunk_ranges)} [{start:>5}:{end:>5}] "
                  f"len={chunk_len:>5}  NLL={nll:.3f}  PPL={ppl:>7.1f}  {dt:.2f}s")
            
            all_chunk_results.append({
                "doc_idx": doc_idx,
                "start": start,
                "end": end,
                "length": chunk_len,
                "nll": nll,
                "ppl": ppl,
            })

    # ---- save raw chunk results ----
    raw_path = os.path.join(args.out_dir, "nll_per_chunk.pt")
    torch.save({
        "chunks": all_chunk_results,
        "max_len": args.max_len,
        "min_chunk_size": args.min_chunk_size,
        "ckpt": args.ckpt,
        "tokens_file": args.tokens,
    }, raw_path)
    print(f"\n[save] raw chunk results -> {raw_path} ({len(all_chunk_results)} chunks)")

    # ---- bucketed summary (by chunk LENGTH, not position) ----
    buckets = bucketize_by_length(all_chunk_results, n_buckets=args.n_buckets)

    csv_path = os.path.join(args.out_dir, "bucketed_ppl.csv")
    with open(csv_path, "w") as f:
        f.write("length_min,length_max,n_chunks,mean_nll,std_nll,ppl\n")
        for b in buckets:
            f.write(f"{b['length_min']},{b['length_max']},{b['n_chunks']},"
                    f"{b['mean_nll']:.6f},{b['std_nll']:.6f},{b['ppl']:.6f}\n")
    print(f"[save] bucketed -> {csv_path}")

    # Global stats
    global_nll = sum(c["nll"] for c in all_chunk_results) / len(all_chunk_results)
    
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "max_len": args.max_len,
            "min_chunk_size": args.min_chunk_size,
            "n_docs": len(docs),
            "n_chunks_total": len(all_chunk_results),
            "global_mean_nll": global_nll,
            "global_ppl": math.exp(min(global_nll, 50)),
            "buckets": buckets,
        }, f, indent=2)
    print(f"[save] summary -> {json_path}")

    # ---- plot: PPL vs chunk length ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [(b["length_min"] + b["length_max"]) / 2 for b in buckets]
            ys = [b["ppl"] for b in buckets]
            yerr = [b["std_nll"] for b in buckets] if buckets else None

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label="PPL (± std)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Chunk length (tokens)")
            ax.set_ylabel("Perplexity")
            ax.set_title(f"PPL vs Chunk Length — {os.path.basename(args.ckpt)}\n"
                         f"{len(docs)} docs, {len(all_chunk_results)} independent chunks")
            ax.grid(True, which="both", ls="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            png_path = os.path.join(args.out_dir, "length_vs_ppl.png")
            fig.savefig(png_path, dpi=150)
            print(f"[save] plot -> {png_path}")
        except ImportError:
            print("[plot] matplotlib not available, skipping")


if __name__ == "__main__":
    main()