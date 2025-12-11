#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from multiprocessing import Pool, cpu_count
from typing import List

# -----------------------------------------------
# Detect if a run folder is UNSYNCED
# -----------------------------------------------
def is_unsynced(run_path: str) -> bool:
    """
    A run is UNSYNCED if:
    - It contains wandb metadata,
    - AND either has no run-state.json,
    - OR run-state.json does not contain `"synced": true`.
    """

    metadata_file = os.path.join(run_path, "wandb-metadata.json")
    state_file = os.path.join(run_path, "run-state.json")

    # Not a W&B run
    if not os.path.exists(metadata_file):
        return False

    # No state file → definitely unsynced
    if not os.path.exists(state_file):
        return True

    # Load state
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        return not state.get("synced", False)
    except Exception:
        # Corrupted file → treat as unsynced
        return True


# -----------------------------------------------
# Sync a single run
# -----------------------------------------------
def sync_run(run_path: str):
    print(f"[SYNC] {run_path}")
    try:
        subprocess.run(
            [sys.executable, "-m", "wandb", "sync", run_path],
            check=False
        )
    except Exception as e:
        print(f"[ERROR] Sync failed for {run_path}: {e}")


# -----------------------------------------------
# Main function
# -----------------------------------------------
def sync_wandb_runs(base_dir="wandb", last_n: int | None = None, num_workers: int | None = None):
    """
    Sync only UNSYNCED runs under `wandb/`.

    Args:
        base_dir: directory containing W&B run folders
        last_n: if set, sync only the last N unsynced runs (sorted by creation time)
        num_workers: number of parallel worker processes (default = CPU count)
    """

    if not os.path.exists(base_dir):
        print(f"No such directory: {base_dir}")
        return

    # All subfolders in wandb/
    all_runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    # Keep only unsynced runs
    unsynced = [r for r in all_runs if is_unsynced(r)]

    # Sort by folder modification time (approx chronological order)
    unsynced.sort(key=lambda p: os.path.getmtime(p))

    print(f"Found {len(unsynced)} UNSYNCED run(s).")

    if last_n is not None:
        unsynced = unsynced[-last_n:]
        print(f"→ Keeping only the last {last_n} run(s).")

    if not unsynced:
        print("Nothing to sync.")
        return

    # Number of worker processes
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    print(f"Syncing with {num_workers} parallel workers...\n")

    # Multiprocessing pool
    with Pool(processes=num_workers) as pool:
        pool.map(sync_run, unsynced)

    print("\nAll requested runs processed.")


# -----------------------------------------------
# CLI
# -----------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="wandb",
                        help="Directory containing wandb run folders.")
    parser.add_argument("--last", type=int, default=None,
                        help="Only sync the last N runs.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers.")

    args = parser.parse_args()

    sync_wandb_runs(
        base_dir=args.dir,
        last_n=args.last,
        num_workers=args.workers
    )
