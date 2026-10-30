"""Prepare BabyLM 2026 Strict-Small for karpathy/nanoGPT.

Requires: pip install huggingface_hub tokenizers numpy

Outputs in the directory containing this script:
    raw/train/*.train.txt    Downloaded training corpus
    raw/val/*.dev          Downloaded development corpus
    tokenizer.json         Custom 8192-token byte-level BPE tokenizer
    train.bin              Packed uint16 training tokens
    val.bin                Packed uint16 validation tokens
    meta.pkl               nanoGPT training metadata
    stats.json             Counts and provenance

Run on an internet-connected machine; copy tokenizer.json, train.bin,
val.bin and meta.pkl to data/babylm/ on Jean-Zay afterwards.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np

# Set Hugging Face's cache before importing huggingface_hub.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from huggingface_hub import list_repo_files, snapshot_download
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
TRAIN_REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
DEV_REPO = "BabyLM-community/BabyLM-dev"
VOCAB_SIZE = 8192
MIN_FREQUENCY = 2
EOT_TOKEN = "<|endoftext|>"

# Set BABYLM_OFFLINE=1 to rebuild the binaries from previously
# downloaded raw files without querying Hugging Face.
OFFLINE = os.environ.get("BABYLM_OFFLINE", "0") == "1"

# Set BABYLM_RETRAIN_TOKENIZER=1 to overwrite tokenizer.json.
RETRAIN_TOKENIZER = os.environ.get("BABYLM_RETRAIN_TOKENIZER", "0") == "1"


# -----------------------------------------------------------
# Download selected text files (without HF datasets conversion)
# -----------------------------------------------------------
def get_source_files(repo_id: str, split: str) -> list[Path]:
    target = ROOT / "raw" / split
    target.mkdir(parents=True, exist_ok=True)

    def relevant(name: str) -> bool:
        base = Path(name).name.lower()
        if split == "train":
            return base.endswith(".train.txt")
        return base.endswith(".dev") or base.endswith(".dev.txt")

    if not OFFLINE:
        candidates = list_repo_files(repo_id, repo_type="dataset")
        selected = sorted(name for name in candidates if relevant(name))
        if not selected:
            raise RuntimeError(
                f"No {split} files found in {repo_id}. "
                f"Available files: {candidates}"
            )

        print(f"Downloading {len(selected)} {split} files from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=selected,
            local_dir=str(target),
        )
        paths = [target / name for name in selected]
    else:
        paths = sorted(p for p in target.rglob("*")
                       if p.is_file() and relevant(p.name))

    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError(
            f"Missing {split} data in {target}. "
            "Run once with internet access, then set BABYLM_OFFLINE=1."
        )
    print(f"{split}: {len(paths)} files")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")
    return paths


# -----------------------------------------------------------
# Train tokenizer on TRAIN files only
# -----------------------------------------------------------
def prepare_tokenizer(train_paths: list[Path], from_pretrained=False) -> Tokenizer:
    if from_pretrained:
        from transformers import AutoTokenizer
        filename = ROOT / "tokenizer_pretrained.json"
        repo = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"

        tokenizer = AutoTokenizer.from_pretrained(repo)
        tokenizer.save_pretrained("data/babylm/tokenizer_official")

        return tokenizer
    
    filename = ROOT / "tokenizer.json"
    if filename.is_file() and not RETRAIN_TOKENIZER:
        print(f"Loading existing tokenizer: {filename}")
        tokenizer = Tokenizer.from_file(str(filename))
        if tokenizer.get_vocab_size() != VOCAB_SIZE:
            raise ValueError(
                "Existing tokenizer vocabulary size differs from VOCAB_SIZE. "
                "Set BABYLM_RETRAIN_TOKENIZER=1 to rebuild it."
            )
        return tokenizer

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("Training byte-level BPE tokenizer (training split only)...")
    tokenizer.train([str(path) for path in train_paths], trainer=trainer)
    tokenizer.save(str(filename))
    print(f"Saved {filename} ({tokenizer.get_vocab_size()} tokens)")
    return tokenizer


# -----------------------------------------------------------
# Tokenize original text to packed uint16 IDs
# -----------------------------------------------------------
def write_split(
    paths: list[Path], tokenizer: Tokenizer, split: str, eot_id: int
) -> dict:
    path_out = ROOT / f"{split}.bin"
    count = 0
    file_stats = {}

    with path_out.open("wb") as out:
        for path in paths:
            # The 10M-word corpus has reasonably small source files
            # (largest around 15 MB): whole-file encoding avoids
            # artificial BPE boundaries at individual newlines.
            text = path.read_text(encoding="utf-8")
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if ids:
                np.asarray(ids, dtype=np.uint16).tofile(out)
            np.asarray([eot_id], dtype=np.uint16).tofile(out)

            file_stats[path.name] = len(ids) + 1
            count += len(ids) + 1
            print(f"  {split}: {path.name}: {len(ids) + 1:,} tokens")

    print(f"Wrote {path_out}: {count:,} tokens, {path_out.stat().st_size / 1e6:.1f} MB")
    return {"tokens": count, "per_source_tokens": file_stats}


if __name__ == "__main__":
    train_files = get_source_files(TRAIN_REPO, "train")
    val_files = get_source_files(DEV_REPO, "val")

    tokenizer = prepare_tokenizer(train_files, from_pretrained=True)
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    assert eot_id is not None
    assert 0 < vocab_size <= np.iinfo(np.uint16).max

    stats = {
        "train_repo": TRAIN_REPO,
        "val_repo": DEV_REPO,
        "vocab_size": vocab_size,
        "eot_token_id": eot_id,
        "train": write_split(train_files, tokenizer, "train", eot_id),
        "val": write_split(val_files, tokenizer, "val", eot_id),
    }

    # nanoGPT/train.py reads meta['vocab_size']; sample.py needs a
    # small modification to load our tokenizer.json (see notes).
    meta = {
        "vocab_size": vocab_size,
        "tokenizer_file": "tokenizer.json",
        "eot_token_id": eot_id,
    }
    with (ROOT / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    (ROOT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Check full binary streams are readable in nanoGPT's format.
    for split in ("train", "val"):
        arr = np.memmap(ROOT / f"{split}.bin", dtype=np.uint16, mode="r")
        assert len(arr) == stats[split]["tokens"]
        assert int(arr.max()) < vocab_size
        print(f"Verified {split}.bin: {len(arr):,} uint16 token IDs")

    print("Done! Copy train.bin, val.bin, tokenizer.json, meta.pkl to data/babylm/.")
