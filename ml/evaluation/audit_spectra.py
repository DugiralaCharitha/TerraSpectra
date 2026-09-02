from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "data_audit"

# Three example classes. We will inspect three cubes from each class.
LABELS_TO_CHECK = [0, 1, 2]
SAMPLES_PER_CLASS = 3


def main():
    samples_by_label = defaultdict(list)

    with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            samples_by_label[int(row["label"])].append(row["id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))

    for plot_index, label in enumerate(LABELS_TO_CHECK, start=1):
        filenames = samples_by_label[label][:SAMPLES_PER_CLASS]

        plt.subplot(len(LABELS_TO_CHECK), 1, plot_index)

        for filename in filenames:
            cube = np.load(CUBES_DIR / filename).astype(np.float32)

            # Average all pixels: one value for each of the 125 spectral bands.
            spectrum = cube.mean(axis=(0, 1))

            # Compare spectral shape rather than absolute brightness.
            spectrum = spectrum / max(float(spectrum.mean()), 1e-8)

            plt.plot(spectrum, linewidth=1.5, label=filename)

        plt.title(f"Class {label}: normalized mean spectra")
        plt.xlabel("Spectral band (0–124)")
        plt.ylabel("Relative intensity")
        plt.legend(fontsize=7)

        print(f"Class {label}:")
        for filename in filenames:
            print(f"  {filename}")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "spectral_audit_classes_0_1_2.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"\nSaved chart: {output_path}")


if __name__ == "__main__":
    main()