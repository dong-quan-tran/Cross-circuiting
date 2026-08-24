import re
import csv
import argparse
from pathlib import Path

MODELS = ["DF", "TikTok", "VarCNN", "RF"]
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1-score"]

BASE_DIR = Path(".")
LOGS_BASE = BASE_DIR / "logs_multipage"


def parse_metrics_from_log(log_path: Path):
    """
    Parse the last metrics dict line from a *_test.log file.

    Expected line shape like:
      {'Accuracy': 0.0131, 'Precision': 0.0015, 'Recall': 0.0122, 'F1-score': 0.0016}
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Find all occurrences of dict-like patterns
    matches = re.findall(r"\{[^}]*\}", text)
    if not matches:
        return None

    # Use the last dict in the file (final test metrics)
    last = matches[-1]

    # Convert single quotes to double quotes so we can parse with json
    import json
    try:
        jtxt = last.replace("'", '"')
        data = json.loads(jtxt)
    except Exception:
        # Fallback: try to parse key:value pairs manually
        data = {}
        for part in last.strip("{}").split(","):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            k = k.strip().strip("'\"")
            v = v.strip()
            try:
                data[k] = float(v)
            except ValueError:
                pass

    metrics = {}
    for k in METRIC_KEYS:
        val = data.get(k)
        if val is None:
            metrics[k] = ""
        else:
            try:
                metrics[k] = float(val)
            except (TypeError, ValueError):
                metrics[k] = ""
    return metrics


def aggregate_multipage(tag: str):
    """
    Aggregate multi-page test metrics for a given dataset tag from:
      logs_multipage/<TAG>/<MODEL>_test.log
    """
    logs_dir = LOGS_BASE / tag
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    rows = []

    for model in MODELS:
        log_path = logs_dir / f"{model}_test.log"
        if not log_path.is_file():
            print(f"[WARN] Missing {log_path}")
            continue

        metrics = parse_metrics_from_log(log_path)
        if metrics is None:
            print(f"[WARN] Could not parse metrics from {log_path}")
            continue

        row = {"tag": tag, "model": model}
        row.update(metrics)
        rows.append(row)

    return rows


def write_csv(rows, out_path: Path):
    fieldnames = ["tag", "model"] + METRIC_KEYS
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-page *_test.log metrics into a CSV summary."
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Dataset tag, e.g. CW_mix_K4_deltat0p01_N3",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help=(
            "Output CSV path. If not provided, defaults to "
            "multipage_summary_<TAG>.csv in the current directory."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tag = args.tag

    if args.out_csv is not None:
        out_csv = Path(args.out_csv)
    else:
        out_csv = BASE_DIR / f"multipage_summary_{tag}.csv"

    rows = aggregate_multipage(tag)
    if not rows:
        print(f"No aggregated results for tag={tag}")
        return

    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
