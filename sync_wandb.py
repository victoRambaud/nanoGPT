#!/usr/bin/env python3
import os
import sys
import subprocess


def sync_wandb_runs(base_dir="wandb"):
    """
    Sync all offline W&B runs stored under `wandb/`.

    Looks for folders named 'offline-run-*' and syncs each via:
        python -m wandb sync <run_path>
    """

    if not os.path.exists(base_dir):
        print(f"No wandb directory found at: {base_dir}")
        return

    runs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("offline-run")
    ]

    if not runs:
        print("No offline W&B runs found to sync.")
        return

    print(f"Found {len(runs)} offline runs to sync.\n")

    for run in runs:
        print(f"=== Syncing: {run} ===")
        try:
            subprocess.run([sys.executable, "-m", "wandb", "sync", run], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to sync {run}: {e}")
        print()

    print("Done syncing all offline runs.")


if __name__ == "__main__":
    sync_wandb_runs()
