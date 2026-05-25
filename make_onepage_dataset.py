import numpy as np
import os
import argparse


np.random.seed(2024)


def make_onepage_dataset(in_path, out_path, mon_label, n_neg_per_pos=1):
    """
    Create a balanced one-page dataset from a defended CW dataset.

    For the monitored page (mon_label):
      - All instances of that page become positive (y=1)
      - Randomly sample an equal number of negative instances (y=0)

    Saves directly to out_path (a .npz file).
    """
    d = np.load(in_path)
    X = d["X"]
    y = d["y"]

    pos_mask = (y == mon_label)
    neg_mask = ~pos_mask

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]

    n_pos = len(X_pos)
    n_neg_sample = n_pos * n_neg_per_pos

    if n_pos == 0:
        raise ValueError(f"No positive samples found for monitored label {mon_label}")

    if n_neg_sample > len(X_neg):
        raise ValueError(
            f"Requested {n_neg_sample} negatives but only {len(X_neg)} available"
        )

    neg_indices = np.random.choice(len(X_neg), size=n_neg_sample, replace=False)
    X_neg_sampled = X_neg[neg_indices]

    X_out = np.concatenate([X_pos, X_neg_sampled], axis=0)
    y_out = np.concatenate([
        np.ones(n_pos, dtype=np.int64),
        np.zeros(n_neg_sample, dtype=np.int64)
    ])

    perm = np.random.permutation(len(X_out))
    X_out = X_out[perm]
    y_out = y_out[perm]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X_out, y=y_out)

    print(f"Page {mon_label}: {n_pos} pos + {n_neg_sample} neg = {len(X_out)} total")
    print(f"Saved to {out_path}")
    return out_path


def infer_prefix_from_input(in_path):
    """
    Infer a clean output prefix from the defended dataset filename.

    Example:
      datasets/CW_tamaraw_padl1_pin0p005_pout0p005_L0_G0.npz
    becomes:
      CW_tam_padl1_pin0p005_pout0p005_L0_G0
    """
    base = os.path.basename(in_path)

    if not base.endswith(".npz"):
        raise ValueError(f"Input path does not end with .npz: {in_path}")

    stem = base[:-4]  # remove .npz

    if stem.startswith("CW_tamaraw_"):
        stem = "CW_tam_" + stem[len("CW_tamaraw_"):]

    return stem


def make_all_onepage_datasets(in_path, out_dir, n_neg_per_pos=1, prefix=None):
    """
    Create one balanced one-page dataset for every label in the input dataset.

    Output structure:
      out_dir/
        <prefix>_page0.npz
        <prefix>_page1.npz
        ...
    """
    d = np.load(in_path)
    y = d["y"]

    labels = sorted(np.unique(y).tolist())

    if prefix is None:
        prefix = infer_prefix_from_input(in_path)

    print(f"Found {len(labels)} unique labels")
    print(f"Generating one-page datasets under: {out_dir}")
    print(f"Using filename prefix: {prefix}")

    generated = []

    os.makedirs(out_dir, exist_ok=True)

    for label in labels:
        out_name = f"{prefix}_page{label}.npz"
        out_path = os.path.join(out_dir, out_name)

        make_onepage_dataset(
            in_path=in_path,
            out_path=out_path,
            mon_label=int(label),
            n_neg_per_pos=n_neg_per_pos
        )
        generated.append(out_path)

    print(f"Generated {len(generated)} one-page datasets")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_path",
        type=str,
        default="datasets/CW_tamaraw.npz",
        help="Path to defended CW dataset (.npz)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="datasets/CW_tamaraw_onepage/onepage.npz",
        help="Output path for a single one-page dataset"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets/CW_tamaraw_onepage_all",
        help="Output directory for all one-page datasets"
    )
    parser.add_argument(
        "--mon_label",
        type=int,
        default=0,
        help="Monitored label for single-page generation"
    )
    parser.add_argument(
        "--n_neg_per_pos",
        type=int,
        default=1,
        help="Number of negative samples per positive sample"
    )
    parser.add_argument(
        "--all_pages",
        action="store_true",
        help="Generate one-page datasets for all labels in the dataset"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix for all-pages mode"
    )

    args = parser.parse_args()

    if args.all_pages:
        make_all_onepage_datasets(
            in_path=args.in_path,
            out_dir=args.out_dir,
            n_neg_per_pos=args.n_neg_per_pos,
            prefix=args.prefix
        )
    else:
        make_onepage_dataset(
            in_path=args.in_path,
            out_path=args.out_path,
            mon_label=args.mon_label,
            n_neg_per_pos=args.n_neg_per_pos
        )
