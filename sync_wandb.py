#!/usr/bin/env python3
import os
import sys
import json
import subprocess


def is_unsynced(run_path: str) -> bool:
    """
    A run is considered UNSYNCED if:
    - It contains wandb run files
    - AND does NOT contain a run-state.json
      OR run-state.json does NOT contain `"synced": true`
    """

    state_file = os.path.join(run_path, "run-state.json")
    metadata_file = os.path.join(run_path, "wandb-metadata.json")

    # If no metadata file exists -> not a run folder
    if not os.path.exists(metadata_file):
        return False

    # No run-state.json -> definitely unsynced
    if not os.path.exists(state_file):
        return True

    # Load run-state.json and check if it is fully synced
    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        # wandb marks synced runs with "synced": true inside run-state.json
        return not state.get("synced", False)

    except Exception:
        # corrupted state file -> treat as unsynced
        return True


def sync_wandb_runs(base_dir="wandb"):
    """
    Sync only unsynced W&B runs.
    """

    if not os.path.exists(base_dir):
        print(f"No such directory: {base_dir}")
        return

    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    unsynced = [d for d in run_dirs if is_unsynced(d)]

    print(f"Found {len(unsynced)} unsynced run(s).")
    if not unsynced:
        return

    for run in unsynced:
        print(f"=== Syncing: {run} ===")
        subprocess.run([sys.executable, "-m", "wandb", "sync", run])
        print()

    print("All unsynced runs processed.")


if __name__ == "__main__":
    sync_wandb_runs()
