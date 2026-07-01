# make_onepage_mixed.py
#
# Build one-page datasets from a cross-circuit mixed dataset produced by
# mix_cross_circuit.py.
#
# Input .npz (from mix_cross_circuit.py):
#   X      : [M, T] mixed CW sequences
#   groups : [M, K] original site labels for the K circuits in each pool
#
# Output .npz (same format as make_onepage_dataset.py output):
#   X : [M_out, T] mixed CW sequences
#   y : [M_out]    1 if pool contains monitored page, 0 otherwise
#
# One-page label definition:
#   positive (y=1): pool contains at least one circuit visiting monitored page P
#   negative (y=0): no circuit in the pool visits P

import numpy as np
import os
import argparse


np.random.seed(2024)


def make_onepage_mixed(in_path, out_path, mon_label, n_neg_per_pos=1, seed=2024):
    """
    Create a balanced one-page dataset from a mixed CW dataset.

    For monitored page P (mon_label):
      - Positives: mixed flows where any circuit in the pool visited P
      - Negatives: mixed flows where no circuit in the pool visited P
      - Balanced: n_neg_per_pos negatives per positive

    Saves to out_path (.npz) with keys X (sequences) and y (binary labels).
    """
    d = np.load(in_path)
    X = d["X"]
    groups = d["groups"]  # shape [M, K]

    mon_label = int(mon_label)

    # Label each mixed flow
    contains_mon = np.any(groups == mon_label, axis=1)  # [M] bool

    pos_mask = contains_mon
    neg_mask = ~contains_mon

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]

    n_pos = len(X_pos)
    n_neg_sample = n_pos * n_neg_per_pos

    if n_pos == 0:
        raise ValueError(
            f"No positive pools found for monitored label {mon_label}. "
            f"Check that this label exists in the source dataset."
        )

    if n_neg_sample > len(X_neg):
        raise ValueError(
            f"Requested {n_neg_sample} negatives but only {len(X_neg)} available "
            f"for page {mon_label}. Reduce n_neg_per_pos or generate more mixed flows."
        )

    rng = np.random.default_rng(seed)
    neg_indices = rng.choice(len(X_neg), size=n_neg_sample, replace=False)
    X_neg_sampled = X_neg[neg_indices]

    X_out = np.concatenate([X_pos, X_neg_sampled], axis=0)
    y_out = np.concatenate([
        np.ones(n_pos, dtype=np.int64),
        np.zeros(n_neg_sample, dtype=np.int64),
    ])

    perm = rng.permutation(len(X_out))
    X_out = X_out[perm]
    y_out = y_out[perm]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X_out, y=y_out)

    print(f"Page {mon_label}: {n_pos} pos + {n_neg_sample} neg = {len(X_out)} total")
    print(f"Saved to {out_path}")
    return out_path


def infer_prefix_from_input(in_path):
    """
    Infer a clean output prefix from the mixed dataset filename.

    Example:
      datasets/CW_mix_K4_deltat0p01_N10.npz
    becomes:
      CW_mix_K4_deltat0p01_N10
    """
    base = os.path.basename(in_path)
    if not base.endswith(".npz"):
        raise ValueError(f"Input path does not end with .npz: {in_path}")
    return base[:-4]


def make_all_onepage_mixed(in_path, out_dir, n_neg_per_pos=1, prefix=None, seed=2024):
    """
    Create one balanced one-page dataset for every unique label present in
    the groups array of the mixed dataset.

    Output structure:
      out_dir/
        <prefix>_page0.npz
        <prefix>_page1.npz
        ...
    """
    d = np.load(in_path)
    groups = d["groups"]  # [M, K]

    # All unique site labels that appear across any pool position
    all_labels = sorted(np.unique(groups).tolist())

    if prefix is None:
        prefix = infer_prefix_from_input(in_path)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(all_labels)} unique labels in mixed dataset")
    print(f"Generating one-page datasets under: {out_dir}")
    print(f"Using filename prefix: {prefix}")

    generated = []

    for label in all_labels:
        out_name = f"{prefix}_page{label}.npz"
        out_path = os.path.join(out_dir, out_name)

        try:
            make_onepage_mixed(
                in_path=in_path,
                out_path=out_path,
                mon_label=int(label),
                n_neg_per_pos=n_neg_per_pos,
                seed=seed,
            )
            generated.append(out_path)
        except ValueError as e:
            print(f"Skipping page {label}: {e}")

    print(f"Generated {len(generated)} one-page datasets")
    return generated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_path",
        type=str,
        default="datasets/CW_mix_K4_deltat0p01_N10.npz",
        help="Path to mixed dataset (.npz) with X and groups",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output path for a single one-page dataset (used without --all_pages)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for all one-page datasets (used with --all_pages)",
    )
    parser.add_argument(
        "--mon_label",
        type=int,
        default=0,
        help="Monitored label for single-page generation",
    )
    parser.add_argument(
        "--n_neg_per_pos",
        type=int,
        default=1,
        help="Number of negative samples per positive sample",
    )
    parser.add_argument(
        "--all_pages",
        action="store_true",
        help="Generate one-page datasets for all labels in the mixed dataset",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix for all-pages mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="RNG seed",
    )

    args = parser.parse_args()

    if args.all_pages:
        # Default out_dir: same directory as input, named after the input file
        out_dir = args.out_dir
        if out_dir is None:
            stem = infer_prefix_from_input(args.in_path)
            out_dir = os.path.join(os.path.dirname(args.in_path), stem + "_pages")

        make_all_onepage_mixed(
            in_path=args.in_path,
            out_dir=out_dir,
            n_neg_per_pos=args.n_neg_per_pos,
            prefix=args.prefix,
            seed=args.seed,
        )

    else:
        out_path = args.out_path
        if out_path is None:
            stem = infer_prefix_from_input(args.in_path)
            out_path = os.path.join(
                os.path.dirname(args.in_path),
                f"{stem}_page{args.mon_label}.npz",
            )

        make_onepage_mixed(
            in_path=args.in_path,
            out_path=out_path,
            mon_label=args.mon_label,
            n_neg_per_pos=args.n_neg_per_pos,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
