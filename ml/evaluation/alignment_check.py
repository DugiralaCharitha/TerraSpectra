"""
Hyperspectral Agriculture Project
CSV ID <-> NPY filename alignment checker

Checks:
1. CSV IDs
2. NPY sample IDs
3. Missing cubes
4. Extra cubes
5. Duplicate IDs
6. Malformed IDs
7. Exact ID-based pairing
8. CSV row order vs NPY order (informational only)

Does NOT modify any dataset files.
"""

from pathlib import Path
import csv
import json
import re
import sys
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\abhi\Downloads\beyond-visible-spectrum-ai-for-agriculture-2025"
)

CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation" / "diagnostic_results"

REPORT_PATH = OUTPUT_DIR / "alignment_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "alignment_summary.json"


# ============================================================
# HELPERS
# ============================================================

def extract_numeric_id(value):
    """
    Convert values such as:
        2451
        sample2451
        sample2451.npy
        "2451"
    into integer 2451.

    Returns None if no valid numeric ID can be extracted.
    """
    if value is None:
        return None

    text = str(value).strip()

    # Remove extension if present
    text = Path(text).stem

    # Prefer sample<number>
    match = re.fullmatch(r"sample(\d+)", text, re.IGNORECASE)

    if match:
        return int(match.group(1))

    # Otherwise accept a pure number
    match = re.fullmatch(r"\d+", text)

    if match:
        return int(match.group(0))

    # Finally, look for sample<number> embedded in a string
    match = re.search(r"sample(\d+)", text, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return None


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPERSPECTRAL AGRICULTURE — ALIGNMENT CHECK")
    print("=" * 70)

    print(f"\nProject root:\n{PROJECT_ROOT}")
    print(f"\nCSV:\n{CSV_PATH}")
    print(f"\nCubes:\n{CUBES_DIR}")
    print(f"\nOutput:\n{OUTPUT_DIR}")

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CSV_PATH.exists():
        print("\nERROR: train_final.csv was not found.")
        print(CSV_PATH)
        sys.exit(1)

    if not CUBES_DIR.exists():
        print("\nERROR: cubes directory was not found.")
        print(CUBES_DIR)
        sys.exit(1)

    # --------------------------------------------------------
    # 1. READ CSV
    # --------------------------------------------------------

    print_section("1. READING CSV")

    with open(CSV_PATH, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []

        print(f"CSV columns: {fieldnames}")

        # Find ID column
        id_column = None

        for column in fieldnames:
            if column.lower() == "id":
                id_column = column
                break

        if id_column is None:
            for column in fieldnames:
                if "id" in column.lower():
                    id_column = column
                    break

        if id_column is None:
            print("\nERROR: Could not find an ID column in CSV.")
            sys.exit(1)

        print(f"Detected ID column: {id_column}")

        csv_rows = list(reader)

    csv_ids = []
    csv_raw_ids = []
    malformed_csv_rows = []

    for row_number, row in enumerate(csv_rows, start=2):
        raw_id = row.get(id_column)

        parsed_id = extract_numeric_id(raw_id)

        csv_raw_ids.append(raw_id)
        csv_ids.append(parsed_id)

        if parsed_id is None:
            malformed_csv_rows.append(
                {
                    "row": row_number,
                    "value": raw_id
                }
            )

    valid_csv_ids = [
        x for x in csv_ids
        if x is not None
    ]

    csv_id_set = set(valid_csv_ids)

    print(f"CSV rows: {len(csv_rows)}")
    print(f"Valid CSV IDs: {len(valid_csv_ids)}")
    print(f"Unique CSV IDs: {len(csv_id_set)}")
    print(f"Malformed CSV IDs: {len(malformed_csv_rows)}")

    # --------------------------------------------------------
    # 2. READ NPY FILES
    # --------------------------------------------------------

    print_section("2. READING NPY FILENAMES")

    npy_files = sorted(CUBES_DIR.glob("*.npy"))

    print(f"NPY files found: {len(npy_files)}")

    cube_ids = []
    cube_files_by_id = {}
    malformed_npy_files = []

    for file_path in npy_files:

        parsed_id = extract_numeric_id(file_path.name)

        if parsed_id is None:
            malformed_npy_files.append(file_path.name)
            continue

        cube_ids.append(parsed_id)

        if parsed_id not in cube_files_by_id:
            cube_files_by_id[parsed_id] = []

        cube_files_by_id[parsed_id].append(file_path.name)

    cube_id_set = set(cube_ids)

    print(f"Valid NPY IDs: {len(cube_ids)}")
    print(f"Unique NPY IDs: {len(cube_id_set)}")
    print(f"Malformed NPY filenames: {len(malformed_npy_files)}")

    # --------------------------------------------------------
    # 3. DUPLICATES
    # --------------------------------------------------------

    print_section("3. DUPLICATE ID CHECK")

    csv_duplicates = sorted(
        {
            x for x in valid_csv_ids
            if valid_csv_ids.count(x) > 1
        }
    )

    cube_duplicates = sorted(
        [
            x
            for x, files in cube_files_by_id.items()
            if len(files) > 1
        ]
    )

    print(f"Duplicate CSV IDs: {len(csv_duplicates)}")
    print(f"Duplicate NPY IDs: {len(cube_duplicates)}")

    if csv_duplicates:
        print("\nDuplicate CSV IDs:")
        print(csv_duplicates)

    if cube_duplicates:
        print("\nDuplicate NPY IDs:")
        for duplicate_id in cube_duplicates:
            print(
                f"  {duplicate_id}: "
                f"{cube_files_by_id[duplicate_id]}"
            )

    # --------------------------------------------------------
    # 4. MISSING / EXTRA
    # --------------------------------------------------------

    print_section("4. ID SET COMPARISON")

    missing_cubes = sorted(
        csv_id_set - cube_id_set
    )

    extra_cubes = sorted(
        cube_id_set - csv_id_set
    )

    print(
        f"CSV IDs without corresponding NPY cube: "
        f"{len(missing_cubes)}"
    )

    print(
        f"NPY cubes without corresponding CSV ID: "
        f"{len(extra_cubes)}"
    )

    if missing_cubes:
        print("\nMissing cubes:")
        print(missing_cubes[:100])

        if len(missing_cubes) > 100:
            print(
                f"... and {len(missing_cubes) - 100} more"
            )

    if extra_cubes:
        print("\nExtra cubes:")
        print(extra_cubes[:100])

        if len(extra_cubes) > 100:
            print(
                f"... and {len(extra_cubes) - 100} more"
            )

    # --------------------------------------------------------
    # 5. EXACT PAIRING TEST
    # --------------------------------------------------------

    print_section("5. CSV ID → NPY PAIRING TEST")

    paired_count = 0
    pairing_failures = []

    for row_number, row in enumerate(csv_rows, start=2):

        raw_id = row.get(id_column)
        sample_id = extract_numeric_id(raw_id)

        if sample_id is None:
            pairing_failures.append(
                {
                    "row": row_number,
                    "id": raw_id,
                    "reason": "Malformed CSV ID"
                }
            )
            continue

        matching_files = cube_files_by_id.get(
            sample_id,
            []
        )

        if len(matching_files) == 1:
            paired_count += 1

        elif len(matching_files) == 0:
            pairing_failures.append(
                {
                    "row": row_number,
                    "id": sample_id,
                    "reason": "No matching NPY cube"
                }
            )

        else:
            pairing_failures.append(
                {
                    "row": row_number,
                    "id": sample_id,
                    "reason": "Multiple NPY files for same ID",
                    "files": matching_files
                }
            )

    print(f"CSV rows successfully paired: {paired_count}")
    print(f"Pairing failures: {len(pairing_failures)}")

    # --------------------------------------------------------
    # 6. ORDER CHECK
    # --------------------------------------------------------

    print_section("6. ORDER CHECK")

    npy_sorted_ids = [
        extract_numeric_id(path.name)
        for path in npy_files
        if extract_numeric_id(path.name) is not None
    ]

    comparable_length = min(
        len(valid_csv_ids),
        len(npy_sorted_ids)
    )

    order_matches = 0

    for i in range(comparable_length):

        if valid_csv_ids[i] == npy_sorted_ids[i]:
            order_matches += 1

    if comparable_length > 0:
        order_match_percentage = (
            order_matches / comparable_length * 100
        )
    else:
        order_match_percentage = 0.0

    print(
        f"Same-position ID matches: "
        f"{order_matches}/{comparable_length}"
    )

    print(
        f"Order match percentage: "
        f"{order_match_percentage:.2f}%"
    )

    print(
        "\nNOTE: Different ordering is NOT a problem "
        "when the data loader matches cubes using IDs."
    )

    # --------------------------------------------------------
    # 7. FINAL STATUS
    # --------------------------------------------------------

    print_section("7. FINAL ALIGNMENT STATUS")

    hard_failures = []

    if malformed_csv_rows:
        hard_failures.append("Malformed CSV IDs")

    if malformed_npy_files:
        hard_failures.append("Malformed NPY filenames")

    if csv_duplicates:
        hard_failures.append("Duplicate CSV IDs")

    if cube_duplicates:
        hard_failures.append("Duplicate NPY IDs")

    if missing_cubes:
        hard_failures.append("CSV IDs missing NPY cubes")

    if pairing_failures:
        hard_failures.append("CSV rows without exactly one cube")

    if len(csv_id_set) != len(cube_id_set):
        # This alone doesn't necessarily mean failure,
        # but missing/extra checks above should explain it.
        pass

    if hard_failures:
        status = "FAIL"

    elif csv_id_set == cube_id_set:
        status = "PASS"

    else:
        status = "WARN"

    print(f"\nALIGNMENT STATUS: {status}")

    if status == "PASS":
        print(
            "\nPASS: Every unique CSV sample ID has exactly "
            "one corresponding NPY cube."
        )

        print(
            "\nThe CSV row order does not matter as long as "
            "your dataset loader uses the ID to locate the cube."
        )

    elif status == "FAIL":
        print("\nPROBLEMS DETECTED:")

        for problem in hard_failures:
            print(f"  - {problem}")

    else:
        print(
            "\nWARNING: CSV and NPY ID sets are not identical."
        )

    # --------------------------------------------------------
    # 8. SAVE JSON
    # --------------------------------------------------------

    summary = {
        "timestamp": datetime.now().isoformat(),

        "project_root": str(PROJECT_ROOT),
        "csv": str(CSV_PATH),
        "cubes_dir": str(CUBES_DIR),

        "csv_rows": len(csv_rows),
        "valid_csv_ids": len(valid_csv_ids),
        "unique_csv_ids": len(csv_id_set),

        "npy_files": len(npy_files),
        "valid_npy_ids": len(cube_ids),
        "unique_npy_ids": len(cube_id_set),

        "malformed_csv_rows": malformed_csv_rows,
        "malformed_npy_files": malformed_npy_files,

        "duplicate_csv_ids": csv_duplicates,
        "duplicate_npy_ids": cube_duplicates,

        "missing_cubes": missing_cubes,
        "extra_cubes": extra_cubes,

        "paired_rows": paired_count,
        "pairing_failures": pairing_failures,

        "order_match_count": order_matches,
        "order_comparison_count": comparable_length,
        "order_match_percentage": order_match_percentage,

        "csv_id_set_equals_npy_id_set": (
            csv_id_set == cube_id_set
        ),

        "status": status
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # 9. SAVE TEXT REPORT
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "HYPERSPECTRAL AGRICULTURE — ALIGNMENT CHECK"
    )

    report_lines.append("=" * 60)

    report_lines.append(
        f"Timestamp: {summary['timestamp']}"
    )

    report_lines.append(
        f"CSV: {CSV_PATH}"
    )

    report_lines.append(
        f"Cubes: {CUBES_DIR}"
    )

    report_lines.append("")

    report_lines.append(
        f"CSV rows: {len(csv_rows)}"
    )

    report_lines.append(
        f"Unique CSV IDs: {len(csv_id_set)}"
    )

    report_lines.append(
        f"NPY files: {len(npy_files)}"
    )

    report_lines.append(
        f"Unique NPY IDs: {len(cube_id_set)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Missing cubes: {len(missing_cubes)}"
    )

    report_lines.append(
        f"Extra cubes: {len(extra_cubes)}"
    )

    report_lines.append(
        f"Duplicate CSV IDs: {len(csv_duplicates)}"
    )

    report_lines.append(
        f"Duplicate NPY IDs: {len(cube_duplicates)}"
    )

    report_lines.append(
        f"Malformed CSV IDs: {len(malformed_csv_rows)}"
    )

    report_lines.append(
        f"Malformed NPY files: {len(malformed_npy_files)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Successfully paired rows: {paired_count}"
    )

    report_lines.append(
        f"Pairing failures: {len(pairing_failures)}"
    )

    report_lines.append("")

    report_lines.append(
        f"CSV ID set == NPY ID set: "
        f"{csv_id_set == cube_id_set}"
    )

    report_lines.append(
        f"Same-position order match: "
        f"{order_match_percentage:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        f"FINAL STATUS: {status}"
    )

    if missing_cubes:
        report_lines.append("")
        report_lines.append("MISSING CUBES:")
        report_lines.extend(
            str(x) for x in missing_cubes
        )

    if extra_cubes:
        report_lines.append("")
        report_lines.append("EXTRA CUBES:")
        report_lines.extend(
            str(x) for x in extra_cubes
        )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(report_lines)
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print_section("CHECK COMPLETE")

    print(
        f"\nJSON saved to:\n{SUMMARY_PATH}"
    )

    print(
        f"\nReport saved to:\n{REPORT_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This script verifies ID ↔ filename alignment."
    )

    print(
        "It cannot prove that the LABEL MEANING is correct."
    )

    print(
        "If STATUS = PASS, the next diagnostic should be "
        "the label-permutation test."
    )


if __name__ == "__main__":
    main()