#!/usr/bin/env python3
import csv
import sys
print(sys.executable)
print(sys.version)
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_diffs(csv_path):
    diffs = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row

        for row_idx, row in enumerate(reader, start=2):
            for col_idx, cell in enumerate(row[1:], start=2):
                try:
                    val = float(cell)
                    diffs.append(abs(val))
                except ValueError:
                    print(f"Warning: non-numeric value at row {row_idx}, col {col_idx}: '{cell}'", file=sys.stderr)
                    continue

    return diffs

def get_all_percent_diffs(diff_csv_path, orig_csv_path):
    percent_diffs = []

    with open(diff_csv_path, newline='') as f_diff, open(orig_csv_path, newline='') as f_orig:
        reader_d = csv.reader(f_diff)
        reader_o = csv.reader(f_orig)

        next(reader_d, None)
        next(reader_o, None)

        for row_idx, (row_d, row_o) in enumerate(zip(reader_d, reader_o), start=2):
            for col_idx, (cell_d, cell_o) in enumerate(zip(row_d[1:], row_o[1:]), start=2):
                try:
                    d = float(cell_d)
                    o = float(cell_o)
                    if o != 0:
                        percent_diffs.append(abs(d / o) * 100)
                except ValueError:
                    print(f"Warning: non-numeric value at row {row_idx}, col {col_idx}: '{cell_d}' or '{cell_o}'", file=sys.stderr)
                    continue

    return percent_diffs

def main():
    diff_file = "tm_score_diff_transformer.csv"
    orig_file = "training_data/training_tm_score_matrix.csv"

    diffs = get_all_diffs(diff_file)
    percent_diffs = get_all_percent_diffs(diff_file, orig_file)

    print(f"Total values: {len(diffs)} differences, {len(percent_diffs)} percent differences")
    print(f"Average difference: {sum(diffs)/len(diffs) if diffs else 'N/A'}")
    print(f"Average percent difference: {sum(percent_diffs)/len(percent_diffs) if percent_diffs else 'N/A'}")

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))

    # Absolute Differences
    plt.subplot(1, 2, 1)
    sns.histplot(diffs, bins=40, kde=True, color='skyblue')
    plt.title("Distribution of Absolute Differences")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Count")

    # Percent Differences
    plt.subplot(1, 2, 2)
    sns.histplot(percent_diffs, bins=40, kde=True, color='salmon')
    plt.title("Distribution of Percent Differences")
    plt.xlabel("Percent Difference (%)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
