
"""Prepare BabyLM 2026 Strict-Small for nanoGPT.

Requires:
    pip install huggingface_hub tokenizers numpy

Custom tokenizer (8,192 BPE tokens):
    python data/babyLM/prepare.py

Official BabyLM 2026 Strict-Small tokenizer:
    BABYLM_TOKENIZER=pretrained \
    BABYLM_OUTPUT_DIR=data/babylm_official \
    python data/babyLM/prepare.py

Optional:
    BABYLM_OFFLINE=1
    BABYLM_RETRAIN_TOKENIZER=1

Outputs:
    train.bin
    val.bin
    val_<source>.bin
    tokenizer.json
    meta.pkl
    stats.json

Both modes use tokenizers.Tokenizer.
Source files are encoded without truncation.
"""

import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from huggingface_hub import list_repo_files, snapshot_download
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
)


# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------

TRAIN_REPO = "BabyLM-community/BabyLM-2026-Strict-Small"
DEV_REPO = "BabyLM-community/BabyLM-dev"

PRETRAINED_REPO = (
    "BabyLM-community/"
    "BabyLM-2026-Baseline-GPT2-Strict-Small"
)

VOCAB_SIZE = 8192*2
MIN_FREQUENCY = 2
CUSTOM_EOS = "<|endoftext|>"

TOKENIZER_MODE = os.environ.get(
    "BABYLM_TOKENIZER", "custom"
).lower()

if TOKENIZER_MODE not in {"custom", "pretrained"}:
    raise ValueError(
        "BABYLM_TOKENIZER must be 'custom' or 'pretrained'"
    )

OUT_DIR = Path(
    os.environ.get("BABYLM_OUTPUT_DIR", str(ROOT))
).expanduser().resolve()

OUT_DIR.mkdir(parents=True, exist_ok=True)

OFFLINE = os.environ.get(
    "BABYLM_OFFLINE", "0"
) == "1"

RETRAIN_TOKENIZER = False


# -----------------------------------------------------------
# Download training and validation files
# -----------------------------------------------------------

def get_source_files(repo_id: str, split: str) -> list[Path]:

    target = ROOT / "raw" / split
    target.mkdir(parents=True, exist_ok=True)

    def relevant(name: str) -> bool:
        basename = Path(name).name.lower()

        if split == "train":
            return basename.endswith(".train.txt")

        return basename.endswith((".dev", ".dev.txt"))

    if OFFLINE:
        paths = sorted(
            p for p in target.rglob("*")
            if p.is_file() and relevant(p.name)
        )

    else:
        candidates = list_repo_files(
            repo_id,
            repo_type="dataset",
        )

        selected = sorted(
            name for name in candidates
            if relevant(name)
        )

        if not selected:
            raise RuntimeError(
                f"No {split} text files found in "
                f"{repo_id}: {candidates}"
            )

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=selected,
            local_dir=str(target),
        )

        paths = [target / name for name in selected]

    if not paths or any(not p.is_file() for p in paths):
        raise RuntimeError(
            f"Missing {split} files in {target}. "
            "Download them once, or unset BABYLM_OFFLINE."
        )

    print(f"{split}: {len(paths)} source files")

    for path in paths:
        print(" ", path.relative_to(ROOT))

    return paths


# -----------------------------------------------------------
# Prepare tokenizer
# -----------------------------------------------------------

def prepare_tokenizer(
    train_paths: list[Path],
) -> tuple[Tokenizer, str, Path]:

    """
    Both branches return tokenizers.Tokenizer.

    This avoids API differences between
    transformers.PreTrainedTokenizerFast
    and tokenizers.Tokenizer.
    """

    if TOKENIZER_MODE == "pretrained":

        tok_dir = ROOT / "pretrained_tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)

        tok_file = tok_dir / "tokenizer.json"

        if not OFFLINE:
            # Download tokenizer files only,
            # not the pretrained model weights.
            snapshot_download(
                repo_id=PRETRAINED_REPO,
                repo_type="model",
                allow_patterns=[
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
                local_dir=str(tok_dir),
            )

        if not tok_file.is_file():
            raise FileNotFoundError(
                f"Missing {tok_file}. "
                "Download the pretrained tokenizer "
                "or copy tokenizer.json here."
            )

        # Directly load the Rust tokenizer.
        # Do NOT use AutoTokenizer.
        tokenizer = Tokenizer.from_file(str(tok_file))

        # Read the official EOS token.
        config_file = tok_dir / "tokenizer_config.json"

        if config_file.exists():
            cfg = json.loads(
                config_file.read_text(encoding="utf-8")
            )

            eos = cfg.get("eos_token", "</s>")

            if isinstance(eos, dict):
                eos = eos["content"]

        else:
            eos = "</s>"

        print(
            f"Official pretrained tokenizer: "
            f"{PRETRAINED_REPO}"
        )

    else:
        # Custom tokenizer
        tok_file = ROOT / "tokenizer_custom.json"
        legacy_file = ROOT / "tokenizer.json"

        # Reuse the previous custom tokenizer,
        # if available.
        if (
            not tok_file.exists()
            and legacy_file.exists()
            and not RETRAIN_TOKENIZER
        ):
            legacy = Tokenizer.from_file(
                str(legacy_file)
            )

            if (
                legacy.get_vocab_size() == VOCAB_SIZE
                and legacy.token_to_id(CUSTOM_EOS) is not None
            ):
                shutil.copyfile(legacy_file, tok_file)

                print(
                    f"Migrated custom tokenizer "
                    f"to {tok_file}"
                )

        if tok_file.exists() and not RETRAIN_TOKENIZER:

            tokenizer = Tokenizer.from_file(
                str(tok_file)
            )

            if tokenizer.get_vocab_size() != VOCAB_SIZE:
                raise ValueError(
                    "Existing custom tokenizer has "
                    "unexpected vocabulary size."
                )

        else:
            tokenizer = Tokenizer(
                models.BPE(unk_token=None)
            )

            tokenizer.pre_tokenizer = (
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False
                )
            )

            tokenizer.decoder = decoders.ByteLevel()

            trainer = trainers.BpeTrainer(
                vocab_size=VOCAB_SIZE,
                min_frequency=MIN_FREQUENCY,
                special_tokens=[CUSTOM_EOS],
                initial_alphabet=(
                    pre_tokenizers.ByteLevel.alphabet()
                ),
                show_progress=True,
            )

            print(
                "Training 8k byte-level BPE "
                "on training files only..."
            )

            tokenizer.train(
                [str(p) for p in train_paths],
                trainer=trainer,
            )

            tokenizer.save(str(tok_file))

        eos = CUSTOM_EOS

    # -------------------------------------------------------
    # Common tokenizer API
    # -------------------------------------------------------

    # Disable truncation and padding.
    # We tokenize complete source files.
    # nanoGPT handles shorter context windows.
    tokenizer.no_truncation()
    tokenizer.no_padding()

    vocab_size = tokenizer.get_vocab_size(
        with_added_tokens=True
    )

    eos_id = tokenizer.token_to_id(eos)

    if eos_id is None:
        raise ValueError(
            f"EOS token {eos!r} "
            f"is missing from {tok_file}"
        )

    if not 0 <= eos_id < vocab_size <= 65536:
        raise ValueError(
            f"Invalid EOS ID / uint16 vocabulary: "
            f"{eos_id}, {vocab_size}"
        )

    print(
        f"Tokenizer: vocab={vocab_size}, "
        f"EOS={eos!r}, EOS ID={eos_id}"
    )

    return tokenizer, eos, tok_file


# -----------------------------------------------------------
# Extract source name
# -----------------------------------------------------------

def source_name(path: Path) -> str:

    name = path.name

    for suffix in (
        ".dev.txt",
        ".dev",
        ".train.txt",
    ):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]

    raise ValueError(
        f"Unexpected source filename: {name}"
    )


# -----------------------------------------------------------
# Tokenize and write binary files
# -----------------------------------------------------------

def write_split(
    paths: list[Path],
    tokenizer: Tokenizer,
    split: str,
    eos_id: int,
) -> dict:

    """
    Encode complete source files.

    Both tokenizer modes return an Encoding
    with an .ids attribute.

    Save a combined binary and separate
    binaries for each validation source.
    """

    total = 0
    counts = {}
    offsets = {}
    subset_files = {}

    combined = OUT_DIR / f"{split}.bin"

    with combined.open("wb") as out:

        for path in paths:

            text = path.read_text(encoding="utf-8")

            # Same API for both tokenizer modes.
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
            ).ids

            # Add EOS between source files.
            arr = np.asarray(
                ids + [eos_id],
                dtype=np.uint16,
            )

            if (
                arr.size
                and int(arr.max())
                >= tokenizer.get_vocab_size()
            ):
                raise ValueError(
                    f"Out-of-vocabulary IDs "
                    f"in {path.name}"
                )

            # Record offsets within combined binary.
            start = total

            arr.tofile(out)

            total += len(arr)

            counts[path.name] = len(arr)

            offsets[path.name] = {
                "start": start,
                "end": total,
            }

            # -----------------------------------------
            # Individual validation subsets
            # -----------------------------------------

            if split == "val":

                name = source_name(path)

                if name in subset_files:
                    raise ValueError(
                        f"Duplicate validation source "
                        f"{name!r}"
                    )

                filename = f"val_{name}.bin"

                arr.tofile(OUT_DIR / filename)

                subset_files[name] = filename

            print(
                f"  {split}: {path.name}: "
                f"{len(arr):,} tokens"
            )

    print(
        f"Saved {combined}: {total:,} tokens "
        f"({combined.stat().st_size / 1e6:.1f} MB)"
    )

    return {
        "tokens": total,
        "per_source_tokens": counts,
        "per_source_offsets": offsets,
        "subset_files": subset_files,
    }


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():

    train_files = get_source_files(
        TRAIN_REPO, "train"
    )

    val_files = get_source_files(
        DEV_REPO, "val"
    )

    tokenizer, eos, tok_source = prepare_tokenizer(
        train_files
    )

    vocab_size = tokenizer.get_vocab_size(
        with_added_tokens=True
    )

    eos_id = tokenizer.token_to_id(eos)

    # Save the actual tokenizer used.
    tok_dest = OUT_DIR / "tokenizer.json"

    if tok_source.resolve() != tok_dest.resolve():
        shutil.copyfile(tok_source, tok_dest)

    # -------------------------------------------------------
    # Process data
    # -------------------------------------------------------

    stats = {
        "train_repo": TRAIN_REPO,
        "val_repo": DEV_REPO,
        "tokenizer_mode": TOKENIZER_MODE,
        "tokenizer_repo": (
            PRETRAINED_REPO
            if TOKENIZER_MODE == "pretrained"
            else None
        ),
        "vocab_size": vocab_size,
        "eos_token": eos,
        "eot_token_id": eos_id,
        "train": write_split(
            train_files,
            tokenizer,
            "train",
            eos_id,
        ),
        "val": write_split(
            val_files,
            tokenizer,
            "val",
            eos_id,
        ),
    }

    # -------------------------------------------------------
    # nanoGPT metadata
    # -------------------------------------------------------

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_file": "tokenizer.json",
        "tokenizer_mode": TOKENIZER_MODE,
        "tokenizer_repo": stats["tokenizer_repo"],
        "eot_token_id": eos_id,
    }

    with (OUT_DIR / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------
    # Verify validation subsets
    # -------------------------------------------------------

    val = np.memmap(
        OUT_DIR / "val.bin",
        dtype=np.uint16,
        mode="r",
    )

    for path in val_files:

        name = source_name(path)

        part = np.memmap(
            OUT_DIR / stats["val"]["subset_files"][name],
            dtype=np.uint16,
            mode="r",
        )

        offsets = (
            stats["val"]["per_source_offsets"][path.name]
        )

        expected = val[
            offsets["start"]:offsets["end"]
        ]

        if not np.array_equal(part, expected):
            raise AssertionError(
                f"Validation subset mismatch: {name}"
            )

    # -------------------------------------------------------
    # Verify binary files
    # -------------------------------------------------------

    for split in ("train", "val"):

        arr = np.memmap(
            OUT_DIR / f"{split}.bin",
            dtype=np.uint16,
            mode="r",
        )

        assert len(arr) == stats[split]["tokens"]
        assert int(arr.max()) < vocab_size

    print(
        "Done. nanoGPT-compatible binaries "
        "and individual validation subsets verified."
    )


if __name__ == "__main__":
    main()
