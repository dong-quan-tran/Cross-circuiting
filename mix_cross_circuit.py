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
# Added metadata:
#       src_indices       : [M, K] original sample indices used in each pool
#       orig_cells        : [M] total real cells across K original traces
#       mixed_cells       : [M] total emitted cells after mixing (before seq truncation)
#       dummy_cells       : [M] total emitted dummy cells
#       real_cells_out    : [M] total emitted real cells
#       delay_mean        : [M] mean added delay for real cells
#       delay_p50         : [M] median added delay for real cells
#       delay_p95         : [M] 95th percentile added delay for real cells
#       delay_max         : [M] max added delay for real cells
#       orig_duration_max : [M] max duration across the K source traces
#       mixed_duration    : [M] duration of mixed trace
#       bw_overhead       : [M] (mixed_cells / orig_cells) - 1
#       lat_overhead      : [M] (mixed_duration / orig_duration_max) - 1
#
# Later, you can build one-page datasets per monitored page P from this
# by labelling each mixed trace as positive if P appears in groups[m].

import os
import time
import argparse
import numpy as np
from random import Random

DATASIZE = 800


def cw_trace_to_packets(trace):
    packets = []
    for v in trace:
        if v == 0:
            continue
        t = abs(float(v))
        sz = DATASIZE if v > 0 else -DATASIZE
        packets.append([t, sz])

    if not packets:
        packets.append([0.0, DATASIZE])

    packets.sort(key=lambda x: x[0])
    return packets


def packets_to_cw_sequence(packets, seq_len):
    packets = sorted(packets, key=lambda x: x[0])
    seq = np.zeros(seq_len, dtype=np.float64)

    for i, (t, sz) in enumerate(packets):
        if i >= seq_len:
            break
        direction = 1.0 if sz > 0 else -1.0
        seq[i] = direction * float(t)

    return seq


def trace_duration_from_cw(trace):
    nz = trace[trace != 0]
    if len(nz) == 0:
        return 0.0
    times = np.abs(nz.astype(np.float64))
    return float(times.max() - times.min())


def _split_by_direction(packets, circuit_idx):
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
    rng = Random(seed)
    out_packets = []
    delays = []

    if N <= 0 or not cells:
        return out_packets, delays, 0, 0

    cells = sorted(cells, key=lambda x: (x["time"], x["circuit_idx"]))

    future_idx = 0
    queue = []

    n_real_out = 0
    n_dummy_out = 0

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
        if eligible:
            eligible.sort(key=lambda x: (x["time"], x["circuit_idx"], rng.random()))

        real = eligible[:N]
        queue = eligible[N:]

        dummy_sign = 1.0
        if real:
            dummy_sign = 1.0 if real[0]["size"] > 0 else -1.0

        for j in range(N):
            out_time = bucket_start + (j + 0.5) * delta_t / float(N)
            if j < len(real):
                cell = real[j]
                out_packets.append([out_time, cell["size"]])
                delays.append(out_time - cell["time"])
                n_real_out += 1
            else:
                sz = dummy_sign * DATASIZE
                out_packets.append([out_time, sz])
                n_dummy_out += 1

        b += 1

    return out_packets, delays, n_real_out, n_dummy_out


def mix_pool(traces, delta_t, N_out, N_in, seed):
    all_out = []
    all_in = []

    for ci, trace in enumerate(traces):
        pkts = cw_trace_to_packets(trace)
        out_cells, in_cells = _split_by_direction(pkts, circuit_idx=ci)
        all_out.extend(out_cells)
        all_in.extend(in_cells)

    out_packets, out_delays, out_real, out_dummy = _mix_one_direction(
        all_out, delta_t=delta_t, N=N_out, seed=seed + 1
    )
    in_packets, in_delays, in_real, in_dummy = _mix_one_direction(
        all_in, delta_t=delta_t, N=N_in, seed=seed + 2
    )

    mixed = []
    for t, sz in out_packets:
        mixed.append([t, abs(sz)])
    for t, sz in in_packets:
        mixed.append([t, -abs(sz)])

    mixed.sort(key=lambda x: x[0])

    delays = np.array(out_delays + in_delays, dtype=np.float64)
    n_real_out = out_real + in_real
    n_dummy_out = out_dummy + in_dummy

    if len(mixed) > 0:
        mixed_times = np.array([p[0] for p in mixed], dtype=np.float64)
        mixed_duration = float(mixed_times.max() - mixed_times.min())
    else:
        mixed_duration = 0.0

    if delays.size > 0:
        delay_mean = float(np.mean(delays))
        delay_p50 = float(np.percentile(delays, 50))
        delay_p95 = float(np.percentile(delays, 95))
        delay_max = float(np.max(delays))
    else:
        delay_mean = 0.0
        delay_p50 = 0.0
        delay_p95 = 0.0
        delay_max = 0.0

    meta = {
        "n_real_out": int(n_real_out),
        "n_dummy_out": int(n_dummy_out),
        "n_mixed_cells": int(len(mixed)),
        "delay_mean": delay_mean,
        "delay_p50": delay_p50,
        "delay_p95": delay_p95,
        "delay_max": delay_max,
        "mixed_duration": mixed_duration,
    }

    return mixed, meta


def _save_partial(
    out_path,
    upto,
    X_mix,
    groups,
    src_indices,
    orig_cells,
    mixed_cells,
    dummy_cells,
    real_cells_out,
    delay_mean,
    delay_p50,
    delay_p95,
    delay_max,
    orig_duration_max,
    mixed_duration,
    bw_overhead,
    lat_overhead,
):
    partial_path = out_path.replace(".npz", f".partial_{upto}.npz")
    np.savez_compressed(
        partial_path,
        X=X_mix[:upto],
        groups=groups[:upto],
        src_indices=src_indices[:upto],
        orig_cells=orig_cells[:upto],
        mixed_cells=mixed_cells[:upto],
        dummy_cells=dummy_cells[:upto],
        real_cells_out=real_cells_out[:upto],
        delay_mean=delay_mean[:upto],
        delay_p50=delay_p50[:upto],
        delay_p95=delay_p95[:upto],
        delay_max=delay_max[:upto],
        orig_duration_max=orig_duration_max[:upto],
        mixed_duration=mixed_duration[:upto],
        bw_overhead=bw_overhead[:upto],
        lat_overhead=lat_overhead[:upto],
    )
    print(f"[checkpoint] saved partial dataset: {partial_path}")


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
    max_traces=None,
    progress_every=100,
    save_every=None,
):
    rng = np.random.default_rng(seed)

    d = np.load(in_path)
    X = d["X"]
    y = d["y"]

    if max_traces is not None:
        X = X[:max_traces]
        y = y[:max_traces]

    N, T = X.shape

    if seq_len is None:
        seq_len = T

    if num_mixed is None:
        num_mixed = N

    num_mixed = min(num_mixed, N)

    X_mix = np.zeros((num_mixed, seq_len), dtype=np.float64)
    groups = np.zeros((num_mixed, K), dtype=np.int64)
    src_indices = np.zeros((num_mixed, K), dtype=np.int64)

    orig_cells = np.zeros(num_mixed, dtype=np.int64)
    mixed_cells = np.zeros(num_mixed, dtype=np.int64)
    dummy_cells = np.zeros(num_mixed, dtype=np.int64)
    real_cells_out = np.zeros(num_mixed, dtype=np.int64)

    delay_mean = np.zeros(num_mixed, dtype=np.float64)
    delay_p50 = np.zeros(num_mixed, dtype=np.float64)
    delay_p95 = np.zeros(num_mixed, dtype=np.float64)
    delay_max = np.zeros(num_mixed, dtype=np.float64)

    orig_duration_max = np.zeros(num_mixed, dtype=np.float64)
    mixed_duration = np.zeros(num_mixed, dtype=np.float64)

    bw_overhead = np.zeros(num_mixed, dtype=np.float64)
    lat_overhead = np.zeros(num_mixed, dtype=np.float64)

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
    print(f"progress_every  : {progress_every}")
    print(f"save_every      : {save_every}")
    print(f"max_traces      : {max_traces}")

    t_start = time.time()

    for m in range(num_mixed):
        idxs = rng.choice(N, size=K, replace=False)
        traces = [X[i] for i in idxs]
        labels = [int(y[i]) for i in idxs]

        mixed_packets, meta = mix_pool(
            traces,
            delta_t=delta_t,
            N_out=N_out,
            N_in=N_in,
            seed=seed + m,
        )

        X_mix[m] = packets_to_cw_sequence(mixed_packets, seq_len=seq_len)
        groups[m] = labels
        src_indices[m] = idxs

        ocells = sum(int(np.count_nonzero(t)) for t in traces)
        odurs = [trace_duration_from_cw(t) for t in traces]
        odur_max = max(odurs) if len(odurs) > 0 else 0.0

        orig_cells[m] = ocells
        mixed_cells[m] = int(meta["n_mixed_cells"])
        dummy_cells[m] = int(meta["n_dummy_out"])
        real_cells_out[m] = int(meta["n_real_out"])

        delay_mean[m] = float(meta["delay_mean"])
        delay_p50[m] = float(meta["delay_p50"])
        delay_p95[m] = float(meta["delay_p95"])
        delay_max[m] = float(meta["delay_max"])

        orig_duration_max[m] = float(odur_max)
        mixed_duration[m] = float(meta["mixed_duration"])

        bw_overhead[m] = float(mixed_cells[m] / ocells - 1.0) if ocells > 0 else 0.0
        lat_overhead[m] = float(mixed_duration[m] / odur_max - 1.0) if odur_max > 0 else 0.0

        done = m + 1
        if done % progress_every == 0 or done == num_mixed:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (num_mixed - done) / rate if rate > 0 else float("inf")
            mean_bw = float(np.mean(bw_overhead[:done]))
            mean_lat = float(np.mean(lat_overhead[:done]))
            print(
                f"Generated {done}/{num_mixed} | "
                f"elapsed={elapsed/60:.1f} min | "
                f"rate={rate:.2f} traces/s | "
                f"ETA={eta/60:.1f} min | "
                f"mean BW={mean_bw:.4f} | mean Lat={mean_lat:.4f}"
            )

        if save_every is not None and (done % save_every == 0):
            _save_partial(
                out_path, done,
                X_mix, groups, src_indices,
                orig_cells, mixed_cells, dummy_cells, real_cells_out,
                delay_mean, delay_p50, delay_p95, delay_max,
                orig_duration_max, mixed_duration,
                bw_overhead, lat_overhead,
            )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        out_path,
        X=X_mix,
        groups=groups,
        src_indices=src_indices,
        orig_cells=orig_cells,
        mixed_cells=mixed_cells,
        dummy_cells=dummy_cells,
        real_cells_out=real_cells_out,
        delay_mean=delay_mean,
        delay_p50=delay_p50,
        delay_p95=delay_p95,
        delay_max=delay_max,
        orig_duration_max=orig_duration_max,
        mixed_duration=mixed_duration,
        bw_overhead=bw_overhead,
        lat_overhead=lat_overhead,
    )

    print(f"Saved mixed dataset to: {out_path}")
    print("Summary")
    print("-------")
    print(f"Mean BW overhead   : {np.mean(bw_overhead):.6f}")
    print(f"Mean Lat overhead  : {np.mean(lat_overhead):.6f}")
    print(f"Mean delay         : {np.mean(delay_mean):.6f}")
    print(f"Mean p95 delay     : {np.mean(delay_p95):.6f}")

    return out_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_path", type=str, default="datasets/CW.npz")
    parser.add_argument("--out_path", type=str, default="datasets/CW_mix_K4.npz")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--delta_t", type=float, default=0.01)
    parser.add_argument("--N_out", type=int, default=10)
    parser.add_argument("--N_in", type=int, default=10)
    parser.add_argument("--num_mixed", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument(
        "--max_traces",
        type=int,
        default=None,
        help="Only use the first max_traces source traces for quick debugging",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print progress every N generated mixed traces",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Checkpoint partial output every N generated traces",
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
        max_traces=args.max_traces,
        progress_every=args.progress_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
