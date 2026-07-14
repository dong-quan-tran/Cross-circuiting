import csv
from pathlib import Path

BASE_DIR = Path(".")

# List all the per-tag summary files you want to merge
SUMMARY_FILES = [
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L1_G1.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L1_G3.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L1_G5.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L1_G7.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L1_G9.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L3_G1.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L5_G1.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L7_G1.csv",
    "onepage_summary_legacyPadl100_pin0p015_pout0p045_L9_G1.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L1_G1.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L1_G3.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L1_G5.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L1_G7.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L1_G9.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L3_G1.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L5_G1.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L7_G1.csv",
    "onepage_summary_legacyPadl100_pin0p025_pout0p075_L9_G1.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L1_G1.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L1_G3.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L1_G5.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L1_G7.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L1_G9.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L3_G1.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L5_G1.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L7_G1.csv",
    "onepage_summary_legacyPadl100_pin0p02_pout0p06_L9_G1.csv",
]

OUT_CSV = BASE_DIR / "onepage_all_schedules.csv"


def main():
    first_header = None
    rows = []

    for fname in SUMMARY_FILES:
        path = BASE_DIR / fname
        if not path.is_file():
            print(f"WARNING: {fname} not found, skipping")
            continue

        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            if first_header is None:
                first_header = header
            else:
                if header != first_header:
                    print(f"WARNING: header mismatch in {fname}")

            for row in reader:
                if row:  # skip empty lines
                    rows.append(row)

    if first_header is None:
        print("No input files found, nothing written.")
        return

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(first_header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
