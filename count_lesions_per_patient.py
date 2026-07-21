#!/usr/bin/env python3
"""Count lesion instances per patient from NIfTI label files."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def iter_nifti_files(input_dir: Path):
    for pattern in ("*.nii.gz", "*.nii"):
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file():
                yield path


def patient_name_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def count_lesions(path: Path) -> int:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    binary = data > 0
    if not np.any(binary):
        return 0

    labeled, num_features = ndimage.label(binary, structure=np.ones((3, 3, 3)))
    return int(num_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count lesion labels per patient from NIfTI segmentation files")
    parser.add_argument(
        "--input-dir",
        default=r"E:\nnUNet_v2_MAIN_FILE\nnUNet_raw\MM_Data_for_zero_input_analysis\MM_GT_DATA\labelsTr",
        help="Directory containing patient NIfTI label files",
    )
    parser.add_argument("--save-csv", action="store_true", help="Save results to a CSV file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = list(iter_nifti_files(input_dir))
    if not files:
        print(f"No .nii or .nii.gz files found in {input_dir}")
        return

    results = []
    for path in files:
        patient = patient_name_from_path(path)
        count = count_lesions(path)
        results.append((patient, count))

    print("Patient lesion counts:")
    for patient, count in sorted(results, key=lambda item: item[0]):
        print(f"{patient}: {count}")

    total_lesions = sum(count for _, count in results)
    print(f"\nTotal lesion count: {total_lesions}")

    if args.save_csv:
        output_path = input_dir / "lesion_counts_per_patient.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["patient", "lesion_count"])
            writer.writerows(sorted(results, key=lambda item: item[0]))
        print(f"Saved CSV report to {output_path}")


if __name__ == "__main__":
    main()
