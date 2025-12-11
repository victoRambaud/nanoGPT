import os
import sys
import subprocess
import logging


# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sync_all_offline_runs(wandb_root_dir):
    """
    Executes the 'wandb sync' command on the specified root directory
    to upload all unsynced offline runs to the WandB server.
    """
    logging.info(f"Starting WandB synchronization for directory: {wandb_root_dir}")

    if not os.path.isdir(wandb_root_dir):
        logging.error(f"Error: WandB directory not found at {wandb_root_dir}")
        logging.error(
            "Please ensure you are running this script from the project root or provide the correct path."
        )
        return

    # The wandb CLI tool handles the discovery and synchronization of all
    # nested offline runs when pointed at the root directory.
    try:
        # Construct the command
        command = ["wandb", "sync", wandb_root_dir]

        logging.info(f"Executing command: {' '.join(command)}")

        # Execute the command and capture output
        result = subprocess.run(
            command,
            check=False,  # Do not raise an exception for non-zero exit codes (WandB might exit 1 if no unsynced runs)
            capture_output=True,
            text=True,
        )

        # Print the standard output and error output
        logging.info("--- WandB Sync Output ---")
        print(result.stdout)

        if result.stderr and "Error" in result.stderr:
            logging.error("--- WandB Sync Error Output ---")
            print(result.stderr)

        if result.returncode == 0:
            # Note: The 'wandb sync' command inherently checks the status of each run
            # and only uploads data that has not been completely synced yet.
            logging.info(
                "Synchronization process completed successfully. Only unsynced parts were uploaded."
            )
        else:
            logging.warning(
                f"Synchronization process finished with return code {result.returncode}. Check output for details."
            )

    except FileNotFoundError:
        logging.error(
            "Error: 'wandb' command not found. Please ensure Weights & Biases CLI is installed and in your PATH."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during sync: {e}")


if __name__ == "__main__":

    # Check for command line argument for a custom path
    if len(sys.argv) > 1:
        sync_path = sys.argv[1]
    else:
        # Default to the standard location in the current working directory
        sync_path = "wandb"
        logging.info(f"No directory specified. Defaulting to: {sync_path}")

    sync_all_offline_runs(sync_path)
