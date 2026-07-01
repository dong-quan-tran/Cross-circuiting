# mix_cross_circuit.py
#
# Cross-circuit mixing simulator for WF experiments.
#
# Starting point:
#   - Input: datasets/CW.npz (same format as used by run_Tamaraw_CW.py)
#       X: [N, T] CW sequences
#          each entry v is 0 or +/- time, sign = direction, magnitude = timestamp
#       y: [N] integer labels (site IDs)
#
#   - Output: a "mixed" dataset .npz:
#       X: [M, T_out] mixed CW sequences
#       groups: [M, K] original labels for the K circuits in each mixed pool
#
# Later, you can build one-page datasets per monitored page P from this
# by labelling each mixed trace as positive if P appears in groups[m].

import os
import argparse
import math
import numpy as np
from random import Random

DATASIZE = 800  # consistent with CW/Tamaraw encoding (cell size placeholder)


def cw_trace_to_packets(trace):
    """
    Convert one CW sequence (1D numpy array) into a packet list:
      trace[i] = 0                -> no packet
      trace[i] = +t (float > 0)   -> outgoing packet at time t, size +DATASIZE
      trace[i] = -t (float < 0)   -> incoming packet at time t, size -DATASIZE

    Returns:
      packets: list of [time, size] sorted by time.
    """
    packets = []
    for v in trace:
        if v == 0:
            continue
        t = abs(float(v))
        sz = DATASIZE if v > 0 else -DATASIZE
        packets.append([t, sz])

    if not packets:
        # avoid degenerate empty traces
        packets.append([0.0, DATASIZE])

    packets.sort(key=lambda x: x[0])
    return packets


def packets_to_cw_sequence(packets, seq_len):
    """
    Convert a packet list [time, size] back into a CW-style 1D array of length seq_len.

    We encode up to seq_len packets; if there are more, we truncate.
    Each entry is +/-time, sign from size sign, magnitude = time.
    """
    packets = sorted(packets, key=lambda x: x[0])
    seq = np.zeros(seq_len, dtype=np.float64)

    for i, (t, sz) in enumerate(packets):
        if i >= seq_len:
            break
        direction = 1.0 if sz > 0 else -1.0
        seq[i] = direction * float(t)

    return seq


def _split_by_direction(packets, circuit_idx):
    """
    Given packets [[time, size], ...] for one circuit, return two lists:
      out_cells: list of {"time", "circuit_idx", "size"} with size > 0
      in_cells : list of {"time", "circuit_idx", "size"} with size < 0
    """
    out_cells = []
    in_cells = []
    for t, sz in packets:
        cell = {"time": float(t), "circuit_idx": int(circuit_idx), "size": float(sz)}
        if sz > 0:
            out_cells.append(cell)
        else:
            in_cells.append(cell)
    return out_cells, in_cells


def _mix_one_direction(cells, delta_t, N, seed):
    """
    Core mixing for a single direction (all sizes share the same sign).

    Inputs:
      cells   : list of dicts {"time", "circuit_idx", "size"} (all same sign)
      delta_t : bucket width in seconds
      N       : per-bucket capacity (cells per bucket)
      seed    : RNG seed for tie-breaking

    Outputs:
      out_packets: list of [time, size] for this direction
      delays     : list of per-cell added delays (for real cells only)
    """
    rng = Random(seed)
    out_packets = []
    delays = []

    if N <= 0:
        return out_packets, delays

    if not cells:
        return out_packets, delays

    # Sort by time, then circuit_idx (stable); RNG only for exact ties.
    cells = sorted(cells, key=lambda x: (x["time"], x["circuit_idx"]))

    future_idx = 0
    queue = []

    # Start buckets at the time of the first cell.
    t0 = cells[0]["time"]
    b = 0

    while future_idx < len(cells) or queue:
        bucket_start = t0 + b * delta_t
        bucket_end = bucket_start + delta_t

        arrivals = []
        while future_idx < len(cells) and cells[future_idx]["time"] < bucket_end:
            if cells[future_idx]["time"] >= bucket_start:
                arrivals.append(cells[future_idx])
            future_idx += 1

        eligible = queue + arrivals

        # If we have no eligible real cells, we still emit N dummies.
        if eligible:
            # Stable sort + random tie-break on exact time/circuit ties.
            eligible.sort(
                key=lambda x: (x["time"], x["circuit_idx"], rng.random())
            )

        real = eligible[:N]
        queue = eligible[N:]

        # Decide sign for dummy cells: use sign of real if available, otherwise +.
        dummy_sign = 1.0
        if real:
            dummy_sign = 1.0 if real[0]["size"] > 0 else -1.0

        for j in range(N):
            out_time = bucket_start + (j + 0.5) * delta_t / float(N)
            if j < len(real):
                cell = real[j]
                out_packets.append([out_time, cell["size"]])
                delays.append(out_time - cell["time"])
            else:
                # Dummy cell
                sz = dummy_sign * DATASIZE
                out_packets.append([out_time, sz])

        b += 1

    return out_packets, delays


def mix_pool(traces, delta_t, N_out, N_in, seed):
    """
    Mix a pool of K per-circuit CW traces into a single mixed trace.

    Inputs:
      traces: list of 1D numpy arrays (CW traces), all length T
      delta_t: bucket width in seconds
      N_out: per-bucket outgoing capacity
      N_in : per-bucket incoming capacity
      seed : base RNG seed

    Output:
      mixed_packets: list of [time, size] sorted by time, size sign = direction
    """
    all_out = []
    all_in = []

    # Build global outgoing/incoming cell lists with circuit indices.
    for ci, trace in enumerate(traces):
        pkts = cw_trace_to_packets(trace)
        out_cells, in_cells = _split_by_direction(pkts, circuit_idx=ci)
        all_out.extend(out_cells)
        all_in.extend(in_cells)

    out_packets, _ = _mix_one_direction(all_out, delta_t=delta_t, N=N_out, seed=seed + 1)
    in_packets, _ = _mix_one_direction(all_in, delta_t=delta_t, N=N_in, seed=seed + 2)

    # Ensure directions: outgoing positive, incoming negative
    mixed = []
    for t, sz in out_packets:
        mixed.append([t, abs(sz)])  # force positive
    for t, sz in in_packets:
        mixed.append([t, -abs(sz)])  # force negative

    mixed.sort(key=lambda x: x[0])
    return mixed


def build_mixed_dataset(
    in_path,
    out_path,
    K=4,
    delta_t=0.01,
    N_out=10,
    N_in=10,
    num_mixed=None,
    seq_len=5000,
    seed=2024,
):
    """
    Build a generic mixed dataset from a CW dataset.

    Inputs:
      in_path   : path to source CW dataset (.npz) with X, y
      out_path  : output path (.npz)
      K         : pool size (circuits per mixed flow)
      delta_t   : bucket width in seconds
      N_out     : outgoing cells per bucket
      N_in      : incoming cells per bucket
      num_mixed : number of mixed flows to generate (default: len(X))
      seq_len   : length of CW sequence to output (default: 5000)
      seed      : RNG seed

    Output .npz:
      X      : [M, seq_len] mixed CW sequences
      groups : [M, K] original labels (site IDs) per mixed flow
    """
    rng = np.random.default_rng(seed)

    d = np.load(in_path)
    X = d["X"]
    y = d["y"]

    N, T = X.shape

    if seq_len is None:
        seq_len = T

    if num_mixed is None:
        num_mixed = N

    X_mix = np.zeros((num_mixed, seq_len), dtype=np.float64)
    groups = np.zeros((num_mixed, K), dtype=np.int64)

    print("Building mixed dataset")
    print("----------------------")
    print(f"Input path      : {in_path}")
    print(f"Output path     : {out_path}")
    print(f"N (orig traces) : {N}")
    print(f"T (orig length) : {T}")
    print(f"K (pool size)   : {K}")
    print(f"delta_t         : {delta_t}")
    print(f"N_out / N_in    : {N_out} / {N_in}")
    print(f"num_mixed       : {num_mixed}")
    print(f"seq_len         : {seq_len}")

    for m in range(num_mixed):
        idxs = rng.choice(N, size=K, replace=False)
        traces = [X[i] for i in idxs]
        labels = [int(y[i]) for i in idxs]

        mixed_packets = mix_pool(
            traces,
            delta_t=delta_t,
            N_out=N_out,
            N_in=N_in,
            seed=seed + m,
        )

        X_mix[m] = packets_to_cw_sequence(mixed_packets, seq_len=seq_len)
        groups[m] = labels

        if (m + 1) % 500 == 0 or (m + 1) == num_mixed:
            print(f"Generated {m + 1}/{num_mixed} mixed traces")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X_mix, groups=groups)
    print(f"Saved mixed dataset to: {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in_path",
        type=str,
        default="datasets/CW.npz",
        help="Input CW dataset (.npz) with X, y",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="datasets/CW_mix_K4.npz",
        help="Output path for mixed dataset (.npz)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="Number of circuits per mixing pool",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.01,
        help="Bucket width in seconds (e.g., 0.01 = 10 ms)",
    )
    parser.add_argument(
        "--N_out",
        type=int,
        default=10,
        help="Outgoing cells per bucket in mixed flow",
    )
    parser.add_argument(
        "--N_in",
        type=int,
        default=10,
        help="Incoming cells per bucket in mixed flow",
    )
    parser.add_argument(
        "--num_mixed",
        type=int,
        default=None,
        help="Number of mixed flows to generate (default: N original traces)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=5000,
        help="Length of CW sequence to output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="RNG seed",
    )

    args = parser.parse_args()

    build_mixed_dataset(
        in_path=args.in_path,
        out_path=args.out_path,
        K=args.K,
        delta_t=args.delta_t,
        N_out=args.N_out,
        N_in=args.N_in,
        num_mixed=args.num_mixed,
        seq_len=args.seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
