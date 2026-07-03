import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values, color='tab:blue')
    ax.set_xlabel('Vertebra')
    ax.set_ylabel('Number of lesions')
    ax.set_title('Number of lesions per vertebra (C1-L6)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
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
    ax.bar(centers, counts, width=widths, align='center', color='tab:green', edgecolor='black')

    if large_count > 0:
        ax.bar(overflow_x, large_count, width=widths[0] * 0.8, align='center', color='tab:orange', edgecolor='black')
        ax.text(overflow_x, large_count, f'> {int(max_size)}: {large_count}', va='bottom', ha='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, max_size + widths[0])
    else:
        ax.set_xlim(0, max_size)

    ax.set_xlabel('Lesion volume (mm^3)')
    ax.set_ylabel('Number of lesions')
    ax.set_title('Histogram of lesion sizes')
    ax.grid(axis='y', alpha=0.3)

    x_ticks = list(np.linspace(0, max_size, min(6, bins)))
    labels = [str(int(t)) for t in x_ticks]
    if large_count > 0:
        labels[-1] = f'> {int(max_size)}'
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot lesion distribution and size histogram from CSV exports.')
    parser.add_argument('--base', default=r'F:\Code\MM_database_viewer', help='Base directory containing exported CSV files')
    parser.add_argument('--save-plots', action='store_true', help='Save plots as PNG files instead of displaying')
    parser.add_argument('--hist-threshold', type=float, default=5000, help='Maximum lesion size to include in histogram bins; larger lesions are grouped separately')
    parser.add_argument('--hist-bins', type=int, default=30, help='Number of bins for the lesion size histogram')
    args = parser.parse_args()

    vertebra_csv = os.path.join(args.base, 'vertebra_distribution.csv')
    sizes_csv = os.path.join(args.base, 'lesion_sizes.csv')

    if not os.path.isfile(vertebra_csv) or not os.path.isfile(sizes_csv):
        raise FileNotFoundError('Required CSV files not found. Run Get_number_of_lesions_vertebraes.py --save-csv first.')

    vertebra_counts = load_vertebra_distribution(vertebra_csv)
    lesion_sizes = load_lesion_sizes(sizes_csv)

    if args.save_plots:
        plot_vertebra_distribution(vertebra_counts, os.path.join(args.base, 'vertebra_distribution.png'))
        plot_lesion_size_histogram(lesion_sizes, max_size=args.hist_threshold, bins=args.hist_bins,
                                   output_path=os.path.join(args.base, 'lesion_size_histogram.png'))
        print(f'Saved vertebra distribution plot to {os.path.join(args.base, "vertebra_distribution.png")}')
        print(f'Saved lesion size histogram to {os.path.join(args.base, "lesion_size_histogram.png")}')
    else:
        plot_vertebra_distribution(vertebra_counts)
        plot_lesion_size_histogram(lesion_sizes, max_size=args.hist_threshold, bins=args.hist_bins)


if __name__ == '__main__':
    main()
