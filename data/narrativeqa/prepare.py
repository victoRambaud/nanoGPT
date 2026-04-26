"""
prepare.py — Download NarrativeQA, tokenize with the GPT-2 tokenizer
(tiktoken), save as a single .pt of token IDs.

Run this on a machine WITH internet. The output .pt is then copied to the cluster.

Usage (from anywhere — output goes next to this script by default):
    python data/narrativeqa/prepare.py
    python data/narrativeqa/prepare.py --max_tokens 20000000 --min_doc_tokens 16384

Notes:
    - Tokenizer is hard-wired to tiktoken's "gpt2" encoding (vocab size 50257).
    - We pull the `narrativeqa` dataset from HuggingFace and use the *story* field
      (full book/movie scripts), not the Q/A pairs. For a length-extrapolation
      probe you want long contiguous text.
    - We deduplicate stories (the dataset repeats each story across many Q/A rows).
    - We keep only stories long enough to actually probe out to your max context.
    - Output is a list of 1-D LongTensors, one per document, plus metadata.
"""

import argparse
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(SCRIPT_DIR, "narrativeqa_tokens.pt"),
                   help="Output path. Defaults to <script_dir>/narrativeqa_tokens.pt")
    p.add_argument("--split", default="test",
                   help="NarrativeQA split: train/validation/test")
    p.add_argument("--max_tokens", type=int, default=20_000_000,
                   help="Stop after this many total tokens (across all docs)")
    p.add_argument("--min_doc_tokens", type=int, default=16384,
                   help="Drop documents shorter than this (no point probing 16k on a 4k doc)")
    p.add_argument("--max_docs", type=int, default=None,
                   help="Optional cap on number of documents")
    args = p.parse_args()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # GPT-2 BPE has no special "BOS"; encode_ordinary skips special-token handling
    # (matters if the corpus contains the literal string "<|endoftext|>", which
    # encode() would otherwise refuse).
    def encode(text: str):
        return enc.encode_ordinary(text)
    print(f"[tokenizer] tiktoken gpt2 (vocab={enc.n_vocab})")

    print(f"[data] loading narrativeqa split={args.split}")
    from datasets import load_dataset
    ds = load_dataset("narrativeqa", split=args.split)

    seen = set()
    docs = []
    total_tokens = 0
    skipped_short = 0

    for i, row in enumerate(ds):
        story_id = row["document"]["id"]
        if story_id in seen:
            continue
        seen.add(story_id)

        text = row["document"]["text"]
        if not text or not text.strip():
            continue

        ids = encode(text)
        if len(ids) < args.min_doc_tokens:
            skipped_short += 1
            continue

        docs.append(torch.tensor(ids, dtype=torch.long))
        total_tokens += len(ids)
        print(f"[data] doc {len(docs)} (story_id={story_id[:8]}…): {len(ids):,} tokens "
              f"(running total {total_tokens:,})")

        if args.max_docs is not None and len(docs) >= args.max_docs:
            break
        if total_tokens >= args.max_tokens:
            break

    if not docs:
        raise RuntimeError(
            f"No documents passed the min_doc_tokens={args.min_doc_tokens} filter. "
            f"Skipped {skipped_short} short docs."
        )

    print(f"[data] kept {len(docs)} documents, {total_tokens:,} total tokens "
          f"(skipped {skipped_short} too-short docs)")

    payload = {
        "docs": docs,                       # list of LongTensors, variable length
        "split": args.split,
        "tokenizer": "tiktoken-gpt2",
        "min_doc_tokens": args.min_doc_tokens,
        "total_tokens": total_tokens,
    }
    torch.save(payload, args.out)
    print(f"[data] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()