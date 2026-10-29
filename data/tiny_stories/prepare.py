
import os
import math

import numpy as np
import tiktoken

from tqdm import tqdm
from datasets import load_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

num_proc = 8
num_proc_load_dataset = 8

# Set to None to use the entire training dataset
max_train_stories = 250_000
max_val_stories = None

seed = 2357
write_batch_size = 2048

enc = tiktoken.get_encoding("gpt2")

out_dir = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def process(example):
    ids = enc.encode_ordinary(example["text"].strip())
    ids.append(enc.eot_token)

    return {
        "ids": ids,
        "len": len(ids),
    }


if __name__ == "__main__":

    # Load TinyStories, including its official
    # train and validation splits.
    dataset = load_dataset(
        "roneneldan/TinyStories",
        num_proc=num_proc_load_dataset,
    )

    train = dataset["train"]
    val = dataset["validation"]

    # Random, reproducible training subset.
    if max_train_stories is not None:
        train = train.shuffle(seed=seed)
        train = train.select(
            range(min(max_train_stories, len(train)))
        )

    if max_val_stories is not None:
        val = val.select(
            range(min(max_val_stories, len(val)))
        )

    split_dataset = {
        "train": train,
        "val": val,
    }

    # Tokenize
    tokenized = {}

    for split, dset in split_dataset.items():
        tokenized[split] = dset.map(
            process,
            remove_columns=["text"],
            desc=f"Tokenizing {split}",
            num_proc=num_proc,
        )

    # --------------------------------------------------
    # Write nanoGPT-compatible binary files
    # --------------------------------------------------

    for split, dset in tokenized.items():

        lengths = np.asarray(
            dset["len"], dtype=np.int64
        )
        arr_len = int(lengths.sum())

        # Print token-length statistics
        print(f"\n{split}:")
        print(f"  Stories: {len(dset):,}")
        print(f"  Tokens:  {arr_len:,}")
        print(f"  Mean:    {lengths.mean():.1f}")
        print(f"  Median:  {np.median(lengths):.1f}")
        print(f"  P90:     {np.percentile(lengths, 90):.1f}")

        for context in [256, 512, 1024]:
            fraction = np.mean(lengths <= context)
            print(
                f"  <= {context} tokens: "
                f"{fraction:.1%}"
            )

        filename = os.path.join(
            out_dir, f"{split}.bin"
        )

        arr = np.memmap(
            filename,
            dtype=np.uint16,
            mode="w+",
            shape=(arr_len,),
        )

        total_batches = math.ceil(
            len(dset) / write_batch_size
        )

        idx = 0

        for batch in tqdm(
            dset.iter(batch_size=write_batch_size),
            total=total_batches,
            desc=f"Writing {filename}",
        ):

            arr_batch = np.concatenate(
                batch["ids"]
            ).astype(np.uint16, copy=False)

            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        assert idx == arr_len

        arr.flush()
        del arr

        print(f"Saved {filename}")
        print(f"Size: {arr_len * 2 / 1e6:.1f} MB")
