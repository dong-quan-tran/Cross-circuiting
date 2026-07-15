# mix_cross_circuit.py
#
# Cross-circuit mixing simulator for WF experiments.
#
# Parallelized version:
#   - Keeps the SAME mixing logic for each pool/trace
#   - Uses multiple CPU processes to generate mixed traces in parallel
#   - Reproducible: each mixed trace m uses seed = base_seed + m
#
# Input .npz:
#   X: [N, T] CW sequences
#   y: [N] integer labels (site IDs)
#
# Output .npz:
#   X                 : [M, seq_len] mixed CW sequences
#   groups            : [M, K] original labels per mixed flow
#   src_indices       : [M, K] original sample indices per mixed flow
#   orig_cells        : [M]
#   mixed_cells       : [M]
#   dummy_cells       : [M]
#   real_cells_out    : [M]
#   delay_mean        : [M]
#   delay_p50         : [M]
#   delay_p95         : [M]
#   delay_max         : [M]
#   orig_duration_max : [M]
#   mixed_duration    : [M]
#   bw_overhead       : [M]
#   lat_overhead      : [M]

import os
import time
import math
import argparse
import numpy as np
from random import Random
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASIZE = 800

# Globals for worker processes
_GLOBAL_X = None
_GLOBAL_Y = None


def _init_worker(X, y):
    global _GLOBAL_X, _GLOBAL_Y
    _GLOBAL_X = X
    _GLOBAL_Y = y


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
        cell = {
            "time": float(t),
            "circuit_idx": int(circuit_idx),
            "size": float(sz),
        }
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


def _compute_one_mixed(args):
    m, K, delta_t, N_out, N_in, seq_len, seed = args

    X = _GLOBAL_X
    y = _GLOBAL_Y
    N = X.shape[0]

    rng = np.random.default_rng(seed + m)
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

    x_mix = packets_to_cw_sequence(mixed_packets, seq_len=seq_len)

    ocells = sum(int(np.count_nonzero(t)) for t in traces)
    odurs = [trace_duration_from_cw(t) for t in traces]
    odur_max = max(odurs) if len(odurs) > 0 else 0.0

    mixed_cells = int(meta["n_mixed_cells"])
    bw = float(mixed_cells / ocells - 1.0) if ocells > 0 else 0.0
    lat = float(meta["mixed_duration"] / odur_max - 1.0) if odur_max > 0 else 0.0

    return {
        "m": m,
        "X": x_mix,
        "groups": np.array(labels, dtype=np.int64),
        "src_indices": np.array(idxs, dtype=np.int64),
        "orig_cells": int(ocells),
        "mixed_cells": mixed_cells,
        "dummy_cells": int(meta["n_dummy_out"]),
        "real_cells_out": int(meta["n_real_out"]),
        "delay_mean": float(meta["delay_mean"]),
        "delay_p50": float(meta["delay_p50"]),
        "delay_p95": float(meta["delay_p95"]),
        "delay_max": float(meta["delay_max"]),
        "orig_duration_max": float(odur_max),
        "mixed_duration": float(meta["mixed_duration"]),
        "bw_overhead": bw,
        "lat_overhead": lat,
    }


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
    num_workers=None,
    chunksize=8,
):
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

    if num_workers is None:
        num_workers = max(1, min(os.cpu_count() or 1, 24))

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
    print(f"num_workers     : {num_workers}")
    print(f"chunksize       : {chunksize}")

    t_start = time.time()
    done = 0
    next_checkpoint = save_every if save_every is not None else None

    tasks = [
        (m, K, delta_t, N_out, N_in, seq_len, seed)
        for m in range(num_mixed)
    ]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(X, y),
    ) as ex:
        futures = [ex.submit(_compute_one_mixed, task) for task in tasks]

        for fut in as_completed(futures):
            res = fut.result()
            m = res["m"]

            X_mix[m] = res["X"]
            groups[m] = res["groups"]
            src_indices[m] = res["src_indices"]

            orig_cells[m] = res["orig_cells"]
            mixed_cells[m] = res["mixed_cells"]
            dummy_cells[m] = res["dummy_cells"]
            real_cells_out[m] = res["real_cells_out"]

            delay_mean[m] = res["delay_mean"]
            delay_p50[m] = res["delay_p50"]
            delay_p95[m] = res["delay_p95"]
            delay_max[m] = res["delay_max"]

            orig_duration_max[m] = res["orig_duration_max"]
            mixed_duration[m] = res["mixed_duration"]

            bw_overhead[m] = res["bw_overhead"]
            lat_overhead[m] = res["lat_overhead"]

            done += 1

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

            if next_checkpoint is not None and done >= next_checkpoint:
                filled = np.where(orig_cells > 0)[0]
                if len(filled) > 0:
                    upto = int(filled.max()) + 1
                    _save_partial(
                        out_path, upto,
                        X_mix, groups, src_indices,
                        orig_cells, mixed_cells, dummy_cells, real_cells_out,
                        delay_mean, delay_p50, delay_p95, delay_max,
                        orig_duration_max, mixed_duration,
                        bw_overhead, lat_overhead,
                    )
                next_checkpoint += save_every

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
        help="Base RNG seed",
    )
    parser.add_argument(
        "--max_traces",
        type=int,
        default=None,
        help="Only use the first max_traces source traces",
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: min(cpu_count, 24))",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Reserved tuning knob; currently informational only",
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
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
