import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy precomputed feature cache from Google Drive into the Colab runtime.")
    parser.add_argument(
        "--drive_dir",
        required=True,
        help="Path inside Google Drive that contains the cached features (e.g., /content/drive/MyDrive/chromatin_feature_cache)",
    )
    parser.add_argument(
        "--local_dir",
        default="./precomputed_feature_cache",
        help="Destination directory inside the Colab runtime.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the local directory if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    drive_dir = Path(args.drive_dir).expanduser()
    local_dir = Path(args.local_dir)

    if not drive_dir.exists():
        raise FileNotFoundError(f"Drive directory not found: {drive_dir}")
    metadata_path = drive_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected metadata.json inside {drive_dir}, but it was not found.")

    if local_dir.exists():
        if args.overwrite:
            shutil.rmtree(local_dir)
        else:
            print(f"{local_dir} already exists. Skipping copy. Use --overwrite to replace it.")
            return

    shutil.copytree(drive_dir, local_dir)

    metadata = json.loads(metadata_path.read_text())
    print(f"Copied feature cache to {local_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()






