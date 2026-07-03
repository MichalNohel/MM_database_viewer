import os
import re
import argparse
from collections import Counter, defaultdict
import statistics

import numpy as np
import nibabel as nib

PATIENT_DIR_RE = re.compile(r'^Myel_\d{3}(_[ab])?$')

REGION_LABELS = {
    'cervical': range(1, 8),
    'thoracic': range(8, 20),
    'lumbar': range(20, 26),
}


def spine_label_region(label):
    for region, label_range in REGION_LABELS.items():
        if label in label_range:
            return region
    return 'unknown'


def vertebra_label_name(label):
    if label >= 1 and label <= 7:
        return f'C{label}'
    if label >= 8 and label <= 19:
        return f'T{label - 7}'
    if label >= 20 and label <= 24:
        return f'L{label - 19}'
    if label == 25:
        return 'L6'
    return 'unknown'


def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data.astype(np.int32), img.header.get_zooms()


def voxel_volume(zooms):
    return float(np.prod(np.abs(zooms)))


def patients_from_base(base_dir):
    for entry in sorted(os.listdir(base_dir)):
        if PATIENT_DIR_RE.match(entry):
            path = os.path.join(base_dir, entry)
            if os.path.isdir(path):
                yield entry, path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute lesion and vertebra segmentation metrics.')
    parser.add_argument('--base', default=r'F:\TCIA\manifest-1774389300184\MM_NIfTI_Segmentation_release_2',
                        help='Base directory containing Myel_* patient folders')
    parser.add_argument('--save-csv', action='store_true', help='Save per-patient results to CSV files')
    args = parser.parse_args()

    base = args.base
    if not os.path.isdir(base):
        raise FileNotFoundError(f'Base directory does not exist: {base}')

    patients = list(patients_from_base(base))
    print(f'Found {len(patients)} patient directories in {base}')

    total_lesions = 0
    total_vertebrae = 0
    total_spine_labels = set()
    patient_lesion_counts = {}
    lesion_volumes = []
    lesion_region_counter = Counter()
    vertebra_lesion_counts = Counter()
    patient_results = []
    lesion_details = []

    for patient_name, patient_dir in patients:
        print(patient_name)
        spine_path = os.path.join(patient_dir, f'{patient_name}_spine_segmentation.nii.gz')
        lesions_path = os.path.join(patient_dir, f'{patient_name}_lesions_segmentation.nii.gz')
        if not os.path.isfile(spine_path) or not os.path.isfile(lesions_path):
            print(f'SKIP {patient_name}: missing spine or lesions mask')
            continue

        spine_mask, spine_zooms = load_mask(spine_path)
        lesions_mask, lesion_zooms = load_mask(lesions_path)
        if spine_zooms != lesion_zooms:
            print(f'WARNING {patient_name}: inconsistent voxel zooms between spine and lesion masks')

        vx_vol = voxel_volume(lesion_zooms)
        vertebra_labels = np.unique(spine_mask)
        vertebra_labels = vertebra_labels[vertebra_labels > 0]
        vertebra_count = len(vertebra_labels)
        total_vertebrae += vertebra_count
        total_spine_labels.update(vertebra_labels.tolist())

        lesion_labels = np.unique(lesions_mask)
        lesion_labels = lesion_labels[lesion_labels > 0]
        patient_lesion_counts[patient_name] = len(lesion_labels)
        total_lesions += len(lesion_labels)

        patient_region_counts = {
            'cervical': 0,
            'thoracic': 0,
            'lumbar': 0,
            'unknown': 0,
        }
        l6_present = 25 in vertebra_labels

        for label in lesion_labels:
            lesion_voxels = lesions_mask == label
            volume_mm3 = int(np.count_nonzero(lesion_voxels) * vx_vol)
            lesion_volumes.append(volume_mm3)

            spine_overlap = spine_mask[lesion_voxels]
            spine_pts, counts = np.unique(spine_overlap, return_counts=True)
            mask_pts = spine_pts > 0
            spine_pts = spine_pts[mask_pts]
            counts = counts[mask_pts]
            if spine_pts.size == 0:
                region = 'unknown'
                vertebra_label = 0
                vertebra_name = 'unknown'
            else:
                vertebra_label = int(spine_pts[np.argmax(counts)])
                vertebra_name = vertebra_label_name(vertebra_label)
                region = spine_label_region(vertebra_label)
                vertebra_lesion_counts[vertebra_label] += 1
            lesion_region_counter[region] += 1
            patient_region_counts[region] += 1
            lesion_details.append({
                'patient': patient_name,
                'lesion_id': int(label),
                'vertebra_label': vertebra_label,
                'vertebra_name': vertebra_name,
                'region': region,
                'volume_mm3': volume_mm3,
            })

        patient_results.append({
            'patient': patient_name,
            'vertebrae': vertebra_count,
            'lesions': int(len(lesion_labels)),
            'cervical': patient_region_counts['cervical'],
            'thoracic': patient_region_counts['thoracic'],
            'lumbar': patient_region_counts['lumbar'],
            'unknown': patient_region_counts['unknown'],
            'L6': 'yes' if l6_present else 'no',
        })

    if not lesion_volumes:
        print('No lesions found in the dataset. Exiting.')
        raise SystemExit(0)

    mean_volume = statistics.mean(lesion_volumes)
    min_volume = min(lesion_volumes)
    max_volume = max(lesion_volumes)
    lesion_counts_desc = sorted(patient_lesion_counts.items(), key=lambda x: x[1], reverse=True)

    print('\n=== Patient table ===')
    header = ['Patient', 'Vertebrae', 'Lesions', 'Cervical', 'Thoracic', 'Lumbar', 'Unknown', 'L6']
    widths = [12, 10, 8, 8, 9, 6, 8, 4]
    row_format = ' '.join(f'{{:<{w}}}' for w in widths)
    print(row_format.format(*header))
    print('-' * sum(widths))
    for row in sorted(patient_results, key=lambda x: x['patient']):
        print(row_format.format(
            row['patient'],
            row['vertebrae'],
            row['lesions'],
            row['cervical'],
            row['thoracic'],
            row['lumbar'],
            row['unknown'],
            row['L6'],
        ))

    print('\n=== Summary ===')
    print(f'Total segmented patients with both masks: {len(patient_results)}')
    print(f'Total number of segmented lesions: {total_lesions}')
    print(f'Total number of segmented vertebrae: {total_vertebrae}')
    print(f'Total distinct vertebra labels seen: {len(total_spine_labels)}')
    print(f'Mean lesion volume: {mean_volume:.1f} mm^3')
    print(f'Lesion volume range: {min_volume} mm^3 to {max_volume} mm^3')
    print('\nLesions per patient (descending):')
    for patient, count in lesion_counts_desc:
        print(f'  {patient}: {count}')

    print('\nAnatomical lesion distribution:')
    for region in ['cervical', 'thoracic', 'lumbar', 'unknown']:
        print(f'  {region}: {lesion_region_counter[region]}')

    if args.save_csv:
        import csv
        output_dir = os.getcwd()
        print(f'Exporting CSV files to: {output_dir}')
        print(f'lesion_details count: {len(lesion_details)}')
        print(f'vertebra_lesion_counts: {dict(vertebra_lesion_counts)}')
        print(f'lesion_volumes count: {len(lesion_volumes)}')
        
        csv_file = os.path.join(output_dir, 'lesion_patient_distribution.csv')
        fieldnames = ['patient', 'lesions', 'vertebrae', 'cervical', 'thoracic', 'lumbar', 'unknown', 'L6']
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(patient_results, key=lambda x: x['patient']))
        print(f'Wrote patient summary CSV: {csv_file}')

        csv_file2 = os.path.join(output_dir, 'lesion_summary.csv')
        with open(csv_file2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['total_lesions', 'total_vertebrae', 'distinct_spine_labels', 'mean_volume_mm3', 'min_volume_mm3', 'max_volume_mm3'])
            writer.writerow([total_lesions, total_vertebrae, len(total_spine_labels), mean_volume, min_volume, max_volume])
        print(f'Wrote summary CSV: {csv_file2}')

        try:
            csv_file3 = os.path.join(output_dir, 'lesion_details.csv')
            fieldnames2 = ['patient', 'lesion_id', 'vertebra_label', 'vertebra_name', 'region', 'volume_mm3']
            with open(csv_file3, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames2)
                writer.writeheader()
                writer.writerows(lesion_details)
            print(f'Wrote lesion details CSV: {csv_file3}')
        except Exception as e:
            print(f'ERROR writing lesion_details.csv: {e}')

        try:
            csv_file4 = os.path.join(output_dir, 'vertebra_distribution.csv')
            fieldnames3 = ['vertebra_label', 'vertebra_name', 'count']
            with open(csv_file4, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames3)
                writer.writeheader()
                for label in range(1, 26):
                    writer.writerow({
                        'vertebra_label': label,
                        'vertebra_name': vertebra_label_name(label),
                        'count': int(vertebra_lesion_counts[label]),
                    })
            print(f'Wrote vertebra distribution CSV: {csv_file4}')
        except Exception as e:
            print(f'ERROR writing vertebra_distribution.csv: {e}')

        try:
            csv_file5 = os.path.join(output_dir, 'lesion_sizes.csv')
            with open(csv_file5, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['volume_mm3'])
                for volume in lesion_volumes:
                    writer.writerow([volume])
            print(f'Wrote lesion sizes CSV: {csv_file5}')
        except Exception as e:
            print(f'ERROR writing lesion_sizes.csv: {e}')
    print(patient_results)
    print('Done.')
