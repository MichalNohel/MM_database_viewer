import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Unified font sizes for plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

VERTEBRA_LABELS = list(range(1, 26))


def vertebra_label_name(label):
    if 1 <= label <= 7:
        return f'C{label}'
    if 8 <= label <= 19:
        return f'T{label - 7}'
    if 20 <= label <= 24:
        return f'L{label - 19}'
    if label == 25:
        return 'L6'
    return 'unknown'


def load_vertebra_distribution(csv_path):
    counts = {label: 0 for label in VERTEBRA_LABELS}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['vertebra_label'])
            counts[label] = int(row['count'])
    return counts


def load_lesion_sizes(csv_path):
    sizes = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(float(row['volume_mm3']))
    return sizes


def plot_vertebra_distribution(counts, output_path=None):
    labels = [vertebra_label_name(l) for l in VERTEBRA_LABELS]
    values = [counts[l] for l in VERTEBRA_LABELS]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color='tab:blue', edgecolor='black')
    ax.set_xlabel('Vertebra')
    ax.set_ylabel('Number of lesions')
    ax.set_title('Number of lesions per vertebra (C1-L6)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


def plot_lesion_size_histogram(sizes, max_size=10000, bins=30, output_path=None):
    large_count = sum(1 for s in sizes if s > max_size)
    sizes_below = [s for s in sizes if s <= max_size]

    counts, edges = np.histogram(sizes_below, bins=bins, range=(0, max_size))
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    overflow_x = max_size

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, counts, width=widths, align='center', color='tab:blue', edgecolor='black')

    if large_count > 0:
        # overflow count shown as red bar
        ax.bar(overflow_x, large_count, width=widths[0] * 0.8, align='center', color='red', edgecolor='black')
        ax.set_xlim(0, max_size + widths[0])
    else:
        ax.set_xlim(0, max_size)

    ax.set_xlabel('Lesion volume (mm³)')
    ax.set_ylabel('Number of lesions')
    ax.set_title('Histogram of lesion sizes', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    x_ticks = list(np.linspace(0, max_size, min(6, bins)))
    labels = [str(int(t)) for t in x_ticks]
    if large_count > 0:
        labels[-1] = f'> {int(max_size)}'
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()




def load_lesions_per_patient_with_ids(csv_path):
    patient_ids = []
    lesions = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pid = row.get('patient') or row.get('Patient')
            try:
                count = int(row['lesions'])
            except (KeyError, ValueError):
                continue
            patient_ids.append(pid)
            lesions.append(count)
    return patient_ids, lesions


# histogram per-patient removed; keep bar chart visualization only


def plot_lesions_per_patient_bars(patient_ids, lesions_counts, output_path=None):
    # dynamic width: make room for labels if many patients
    fig_width = max(10, len(patient_ids) * 0.12)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    x = np.arange(len(patient_ids))
    ax.bar(x, lesions_counts, color='tab:blue', edgecolor='black')
    # scale label/title sizes so they visually match other plots even for wide figures
    base_label = plt.rcParams.get('axes.labelsize', 12)
    base_title = plt.rcParams.get('axes.titlesize', 14)
    scale = fig_width / 10.0
    label_font = int(max(10, round(base_label * scale)))
    title_font = int(max(12, round(base_title * scale)))
    ax.set_title('Lesions per patient', fontweight='bold', fontsize=title_font)
    ax.set_xticks(x)
    # keep the smaller fontsize for patient IDs as before
    ax.set_xticklabels(patient_ids, rotation=90, fontsize=8)
    ax.set_xlabel('Patient', fontsize=label_font)
    ax.set_ylabel('Number of lesions', fontsize=label_font)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if output_path:
        # save with higher DPI and tight bounding box for better quality
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
    else:
        # increase display DPI for on-screen clarity
        fig.set_dpi(150)
        plt.show()


def compute_and_print_patient_stats(lesions_counts, output_path=None):
    arr = np.array(lesions_counts)
    n = len(arr)
    if n == 0:
        print('No patient data available for statistics.')
        return
    mn = int(arr.min())
    mx = int(arr.max())
    rng = mx - mn
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=0))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    lines = [
        f'Patients: {n}',
        f'Min: {mn}',
        f'Max: {mx}',
        f'Range: {rng}',
        f'Mean: {mean:.2f}',
        f'Median: {median:.2f}',
        f'Std: {std:.2f}',
        f'25th percentile: {q1:.2f}',
        f'75th percentile: {q3:.2f}'
    ]

    for l in lines:
        print(l)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for l in lines:
                f.write(l + '\n')


def main():
    parser = argparse.ArgumentParser(description='Plot lesion distribution and size histogram from CSV exports.')
    parser.add_argument('--base', default=r'F:\Code\MM_database_viewer', help='Base directory containing exported CSV files')
    parser.add_argument('--save-plots', action='store_true', help='Save plots as PNG files instead of displaying')
    parser.add_argument('--hist-threshold', type=float, default=5000, help='Maximum lesion size to include in histogram bins; larger lesions are grouped separately')
    parser.add_argument('--hist-bins', type=int, default=30, help='Number of bins for the lesion size histogram')
    args = parser.parse_args()

    vertebra_csv = os.path.join(args.base, 'vertebra_distribution.csv')
    sizes_csv = os.path.join(args.base, 'lesion_sizes.csv')
    patient_csv = os.path.join(args.base, 'lesion_patient_distribution.csv')

    if not os.path.isfile(vertebra_csv) or not os.path.isfile(sizes_csv) or not os.path.isfile(patient_csv):
        raise FileNotFoundError('Required CSV files not found. Run Get_number_of_lesions_vertebraes.py --save-csv first.')

    vertebra_counts = load_vertebra_distribution(vertebra_csv)
    lesion_sizes = load_lesion_sizes(sizes_csv)
    patient_ids, lesions_per_patient = load_lesions_per_patient_with_ids(patient_csv)

    if args.save_plots:
        plot_vertebra_distribution(vertebra_counts, os.path.join(args.base, 'vertebra_distribution.png'))
        plot_lesion_size_histogram(lesion_sizes, max_size=args.hist_threshold, bins=args.hist_bins,
                                   output_path=os.path.join(args.base, 'lesion_size_histogram.png'))
        # bar chart per individual patient with labels
        plot_lesions_per_patient_bars(patient_ids, lesions_per_patient, output_path=os.path.join(args.base, 'lesions_per_patient_bars.png'))
        # compute and save basic statistics
        compute_and_print_patient_stats(lesions_per_patient, output_path=os.path.join(args.base, 'lesions_per_patient_stats.txt'))
        print(f'Saved vertebra distribution plot to {os.path.join(args.base, "vertebra_distribution.png")}')
        print(f'Saved lesion size histogram to {os.path.join(args.base, "lesion_size_histogram.png")}')
        print(f'Saved lesions-per-patient bars to {os.path.join(args.base, "lesions_per_patient_bars.png")}')
        print(f'Saved lesions-per-patient stats to {os.path.join(args.base, "lesions_per_patient_stats.txt")}')
    else:
        plot_vertebra_distribution(vertebra_counts)
        plot_lesion_size_histogram(lesion_sizes, max_size=args.hist_threshold, bins=args.hist_bins)
        plot_lesions_per_patient_bars(patient_ids, lesions_per_patient)
        compute_and_print_patient_stats(lesions_per_patient)


if __name__ == '__main__':
    main()
