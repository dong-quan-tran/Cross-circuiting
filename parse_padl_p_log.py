import re
import csv
from pathlib import Path

LOG_DIR = Path("logs_tamaraw")
LOG_FILE = LOG_DIR / "padl_p_grid_20260523_205628.log"
OUT_CSV = Path("padl_p_overhead_summary.csv")

# Regex patterns
# Example lines:
#   Running: padL=1, p_in=0.005, p_out=0.005, L=0, G=0
#   Average bandwidth overhead: 110.43%
#   Saved defended dataset to: datasets/CW_tamaraw_padl1_pin0p005_pout0p005_L0_G0.npz

RUN_RE = re.compile(
    r"Running:\s*padL=(?P<padl>\d+),\s*"
    r"p_in=(?P<p_in>[0-9.]+),\s*"
    r"p_out=(?P<p_out>[0-9.]+),\s*"
    r"L=(?P<L>-?\d+),\s*G=(?P<G>-?\d+)"
)

OVERHEAD_RE = re.compile(
    r"Average bandwidth overhead:\s*(?P<overhead>[0-9.]+)%"
)

DATASET_RE = re.compile(
    r"Saved defended dataset to:\s*(?P<dataset>\S+)"
)


def parse_log(path: Path):
    records = []

    current = None  # holds the current run's info until we have all fields

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Match the "Running: padL=..., p_in=..., ..." line
            m_run = RUN_RE.match(line)
            if m_run:
                # If we encounter a new run and still have an incomplete current,
                # you could optionally append it or discard it; here we just reset.
                current = {
                    "padL": int(m_run.group("padl")),
                    "p_in": float(m_run.group("p_in")),
                    "p_out": float(m_run.group("p_out")),
                    "L": int(m_run.group("L")),
                    "G": int(m_run.group("G")),
                    "dataset": None,
                    "overhead_percent": None,
                }
                continue

            # Match the dataset line (after the run)
            m_dataset = DATASET_RE.match(line)
            if m_dataset and current is not None:
                current["dataset"] = m_dataset.group("dataset")
                continue

            # Match the overhead line
            m_over = OVERHEAD_RE.match(line)
            if m_over and current is not None:
                current["overhead_percent"] = float(m_over.group("overhead"))

                # Once we have overhead, we expect we already saw the dataset line.
                # If dataset is still None, we'll still record, but that would be a sign to inspect the log.
                records.append(current)
                current = None

    return records


def write_csv(records, out_path: Path):
    fieldnames = [
        "dataset",
        "padL",
        "p_in",
        "p_out",
        "L",
        "G",
        "overhead_percent",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    records = parse_log(LOG_FILE)

    if not records:
        print(f"No records parsed from {LOG_FILE}")
        return

    write_csv(records, OUT_CSV)
    print(f"Wrote {len(records)} records to {OUT_CSV}")


if __name__ == "__main__":
    main()
