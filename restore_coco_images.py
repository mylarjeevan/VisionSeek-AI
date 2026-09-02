import json
import urllib.request
from pathlib import Path

# Project root = folder containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coco-val-2017-500"
    / "metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coco-val-2017-500"
)

COCO_BASE_URL = "http://images.cocodataset.org/val2017"


def download_image(filename):
    output_path = OUTPUT_DIR / filename

    # Skip images already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists: {filename}")
        return True

    url = f"{COCO_BASE_URL}/{filename}"

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded: {filename}")
        return True

    except Exception as error:
        print(f"FAILED: {filename} -> {error}")
        return False


def main():
    if not METADATA_PATH.exists():
        print(f"ERROR: Metadata not found:\n{METADATA_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    filenames = sorted({
        item["filename"]
        for item in metadata
        if item.get("filename")
    })

    print(f"Images required: {len(filenames)}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("-" * 50)

    successful = 0
    failed = []

    for index, filename in enumerate(filenames, start=1):
        print(f"[{index}/{len(filenames)}] {filename}")

        if download_image(filename):
            successful += 1
        else:
            failed.append(filename)

    print("\n" + "=" * 50)
    print("RESTORE COMPLETE")
    print("=" * 50)
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for filename in failed:
            print(filename)


if __name__ == "__main__":
    main()