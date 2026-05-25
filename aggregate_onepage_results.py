import json
import csv
from pathlib import Path

# Configuration for this run
TAG = "padl1_pin0p005_pout0p005_L0_G0"

# Directories
BASE_DIR = Path(".")
LOGS_DIR = BASE_DIR / "logs"
OUT_CSV = BASE_DIR / f"onepage_summary_{TAG}.csv"

# Models we expect
MODELS = ["DF", "TikTok", "VarCNN", "RF"]

# Metric keys as stored in result.json
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-score"]


def find_result_files_for_tag(tag: str):
    """
    Find all result.json files for a given TAG and model.

    Expected pattern:
      logs/CW_tam_<TAG>_page*/MODEL/result.json
    """
    results = {m: [] for m in MODELS}

    # Pattern: logs/CW_tam_<TAG>_page*
    pattern = f"CW_tam_{tag}_page"
    for page_dir in LOGS_DIR.glob(pattern + "*"):
        if not page_dir.is_dir():
            continue
        # page_dir name is e.g. CW_tam_padl1_pin0p005_pout0p005_L0_G0_page90
        for model in MODELS:
            model_dir = page_dir / model
            result_file = model_dir / "result.json"
            if result_file.is_file():
                results[model].append(result_file)

    return results


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # data is assumed to be a flat dict with metric names as keys
    return {k: float(data[k]) for k in METRIC_KEYS if k in data}


def aggregate_results(tag: str):
    all_results = find_result_files_for_tag(tag)

    rows = []
    for model, files in all_results.items():
        if not files:
            print(f"No result.json files found for model {model}")
            continue

        sums = {k: 0.0 for k in METRIC_KEYS}
        count = 0

        for path in files:
            metrics = load_metrics(path)
            if not metrics:
                continue
            for k in METRIC_KEYS:
                if k in metrics:
                    sums[k] += metrics[k]
            count += 1

        if count == 0:
            continue

        avg = {k: (sums[k] / count) for k in METRIC_KEYS}

        row = {
            "tag": tag,
            "model": model,
            "num_pages": count,
        }
        row.update(avg)
        rows.append(row)

    return rows


def write_csv(rows, out_path: Path):
    fieldnames = ["tag", "model", "num_pages"] + METRIC_KEYS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    rows = aggregate_results(TAG)
    if not rows:
        print(f"No aggregated results for tag={TAG}")
        return
    write_csv(rows, OUT_CSV)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
