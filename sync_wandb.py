#!/usr/bin/env python3
import os
import sys
import subprocess


def sync_run(run_path: str):
    print(f"[SYNC] {run_path}")
    try:
        subprocess.run([sys.executable, "-m", "wandb", "sync", run_path], check=False)
    except Exception as e:
        print(f"[ERROR] Sync failed for {run_path}: {e}")


def sync_all_runs(base_dir="wandb", last_n: int | None = None):
    if not os.path.exists(base_dir):
        print(f"No such directory: {base_dir}")
        return

    # List all run folders inside wandb/
    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not run_dirs:
        print("No run directories found.")
        return

    # Sort by last modification time (oldest first → newest last)
    run_dirs.sort(key=lambda p: os.path.getmtime(p))

    print(f"Found {len(run_dirs)} total runs.")

    if last_n is not None:
        run_dirs = run_dirs[-last_n:]
        print(f"→ Keeping only the last {last_n} run(s).")

    print("Starting sequential sync...\n")

    for run in run_dirs:
        sync_run(run)

    print("\nAll selected runs have been synced.")


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

    args = parser.parse_args()

    sync_all_runs(
        base_dir=args.dir,
        last_n=args.last,
    )
