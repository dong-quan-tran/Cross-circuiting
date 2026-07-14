import json
import csv
import argparse
from pathlib import Path

MODELS = ["DF", "TikTok", "VarCNN", "RF"]
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-score", "TPR", "FPR"]

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"


def page_sort_key(path: Path):
    """
    Sort page directories numerically by trailing _pageX if present.
    """
    name = path.name
    if "_page" in name:
        try:
            return int(name.rsplit("_page", 1)[1])
        except ValueError:
            pass
    return name


def find_result_files_for_tag(tag: str):
    """
    Find all result.json files for a given tag.

    Expected layout:
      logs/CW_mix_<TAG>_page*/MODEL/result.json

    Example:
      logs/CW_mix_K4_deltat0p01_N3_page*/MODEL/result.json
    """
    if not LOGS_DIR.exists():
        raise FileNotFoundError(f"Logs directory not found: {LOGS_DIR}")

    results = {m: [] for m in MODELS}

    page_prefix = f"CW_mix_{tag}_page"
    page_dirs = [p for p in LOGS_DIR.glob(f"{page_prefix}*") if p.is_dir()]
    page_dirs = sorted(page_dirs, key=page_sort_key)

    for page_dir in page_dirs:
        for model in MODELS:
            result_file = page_dir / model / "result.json"
            if result_file.is_file():
                results[model].append(result_file)

    return results


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row = {}
    for k in METRIC_KEYS:
        if k in data:
            try:
                row[k] = float(data[k])
            except (TypeError, ValueError):
                row[k] = None
        else:
            row[k] = None
    return row


def aggregate_results(tag: str):
    all_results = find_result_files_for_tag(tag)

    rows = []
    for model, files in all_results.items():
        if not files:
            print(f"No result.json files found for model {model} (tag={tag})")
            continue

        sums = {k: 0.0 for k in METRIC_KEYS}
        counts = {k: 0 for k in METRIC_KEYS}

        for path in files:
            metrics = load_metrics(path)
            for k in METRIC_KEYS:
                val = metrics.get(k)
                if val is not None:
                    sums[k] += val
                    counts[k] += 1

        row = {
            "tag": tag,
            "model": model,
            "num_pages": len(files),
        }

        for k in METRIC_KEYS:
            if counts[k] > 0:
                row[k] = sums[k] / counts[k]
            else:
                row[k] = ""

        rows.append(row)

    return rows


def write_csv(rows, out_path: Path):
    fieldnames = ["tag", "model", "num_pages"] + METRIC_KEYS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate one-page result.json files into a CSV summary."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Dataset tag, e.g. K4_deltat0p01_N3",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help=(
            "Output CSV path. If not provided, defaults to "
            "onepage_summary_<TAG>.csv in the script directory."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.tag

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        out_csv = BASE_DIR / f"onepage_summary_{tag}.csv"

    rows = aggregate_results(tag)
    if not rows:
        print(f"No aggregated results for tag={tag}")
        return

    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
