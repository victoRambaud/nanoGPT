#!/usr/bin/env python3
import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count


def sync_run(run_path: str):
    print(f"[SYNC] {run_path}")
    try:
        subprocess.run([sys.executable, "-m", "wandb", "sync", run_path], check=False)
    except Exception as e:
        print(f"[ERROR] Sync failed for {run_path}: {e}")


def sync_all_runs(
    base_dir="wandb", last_n: int | None = None, num_workers: int | None = None
):
    if not os.path.exists(base_dir):
        print(f"No such directory: {base_dir}")
        return

    # List all subdirectories in wandb/
    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not run_dirs:
        print("No run directories found.")
        return

    # Sort by modification time (oldest first → newest last)
    run_dirs.sort(key=lambda p: os.path.getmtime(p))

    print(f"Found {len(run_dirs)} total runs.")

    # If last_n specified, keep only the last few
    if last_n is not None:
        run_dirs = run_dirs[-last_n:]
        print(f"→ Keeping only the last {last_n} run(s).")

    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)

    print(f"Syncing using {num_workers} parallel workers...\n")

    with Pool(processes=num_workers) as pool:
        pool.map(sync_run, run_dirs)

    print("\nAll selected runs have been processed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=str,
        default="wandb",
        help="Directory containing wandb run folders.",
    )
    parser.add_argument(
        "--last", type=int, default=None, help="Only sync the last N runs."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel worker processes."
    )

    args = parser.parse_args()

    sync_all_runs(
        base_dir=args.dir,
        last_n=args.last,
        num_workers=args.workers,
    )
