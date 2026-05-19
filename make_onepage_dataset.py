import numpy as np
import os
import argparse


np.random.seed(2024)


def make_onepage_dataset(in_path, out_dir, mon_label, n_neg_per_pos=1):
    """
    Create a balanced one-page dataset from a defended CW dataset.
   
    For the monitored page (mon_label):
      - All instances become positive (y=1)
      - Randomly sample equal number of negative instances (y=0)
    """
    d = np.load(in_path)
    X = d['X']
    y = d['y']
   
    pos_mask = (y == mon_label)
    neg_mask = ~pos_mask
   
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
   
    n_pos = len(X_pos)
    n_neg_sample = n_pos * n_neg_per_pos
   
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
   
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "onepage.npz")
    np.savez_compressed(out_path, X=X_out, y=y_out)
   
    print(f"Page {mon_label}: {n_pos} pos + {n_neg_sample} neg = {len(X_out)} total")
    print(f"Saved to {out_path}")
    return out_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='datasets/CW_tamaraw.npz')
    parser.add_argument('--out_dir', type=str, default='datasets/CW_tamaraw_onepage')
    parser.add_argument('--mon_label', type=int, default=0)
    parser.add_argument('--n_neg_per_pos', type=int, default=1)
    args = parser.parse_args()
   
    make_onepage_dataset(args.in_path, args.out_dir, args.mon_label, args.n_neg_per_pos)