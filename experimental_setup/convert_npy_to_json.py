"""
convert_npy_to_json.py
----------------------
Converts a directory of sample folders (each containing anchor.npy +
candidate1.npy ... candidate9.npy) into a single data.json file that
the labeling tool can load directly.

Expected folder structure:
    data_root/
        sample_0/
            anchor.npy
            candidate1.npy
            candidate2.npy
            ...
            candidate9.npy
        sample_1/
            anchor.npy
            ...
        ...

Usage:
    python convert_npy_to_json.py --data_dir /path/to/data_root --output data.json

    # Or with explicit list of folders:
    python convert_npy_to_json.py --folders sample_0 sample_1 sample_3 --output data.json
"""

import argparse
import json
import os
import sys

import numpy as np


def load_sample(folder_path: str) -> dict:
    """Load anchor + candidates from a single sample folder."""
    anchor_path = os.path.join(folder_path, "anchor.npy")
    if not os.path.exists(anchor_path):
        raise FileNotFoundError(f"anchor.npy not found in {folder_path}")

    anchor = np.load(anchor_path).flatten().tolist()

    candidates = []
    for i in range(1, 10):
        cand_path = os.path.join(folder_path, f"candidate{i}.npy")
        if not os.path.exists(cand_path):
            raise FileNotFoundError(f"candidate{i}.npy not found in {folder_path}")
        candidates.append(np.load(cand_path).flatten().tolist())

    return {
        "name": os.path.basename(folder_path),
        "anchor": anchor,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert .npy sample folders to JSON for the labeling tool.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_dir",
        type=str,
        help="Root directory containing sample_* sub-folders. All sub-folders are included, sorted by name.",
    )
    group.add_argument(
        "--folders",
        nargs="+",
        type=str,
        help="Explicit list of sample folder paths to include (in order).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.json",
        help="Output JSON file path (default: data.json)",
    )
    args = parser.parse_args()

    if args.data_dir:
        root = args.data_dir
        if not os.path.isdir(root):
            sys.exit(f"Error: {root} is not a directory.")
        folders = sorted(
            [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )
    else:
        folders = args.folders

    if not folders:
        sys.exit("Error: No sample folders found.")

    samples = []
    for folder in folders:
        print(f"Loading {folder} ...", end=" ")
        try:
            sample = load_sample(folder)
            samples.append(sample)
            print(f"OK  (anchor len={len(sample['anchor'])}, candidates={len(sample['candidates'])})")
        except FileNotFoundError as e:
            print(f"SKIP — {e}")

    if not samples:
        sys.exit("Error: No valid samples loaded.")

    output_data = {"samples": samples}
    with open(args.output, "w") as f:
        json.dump(output_data, f)

    print(f"\nDone. {len(samples)} sample(s) written to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
