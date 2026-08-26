import argparse
import time
from pathlib import Path
from random import Random

import numpy as np

DATASIZE = 800


def relative_packets_from_cw(trace):
    packets = []

    for value in trace:
        if value == 0:
            continue

        packets.append(
            (
                abs(float(value)),
                DATASIZE if value > 0 else -DATASIZE,
            )
        )

    packets.sort(key=lambda packet: packet[0])
    return packets


def trace_to_live_cells(trace, circuit_index, open_time):
    return [
        {
            "time": float(open_time + relative_time),
            "circuit_index": int(circuit_index),
            "size": float(packet_size),
        }
        for relative_time, packet_size in relative_packets_from_cw(trace)
    ]


def packets_to_cw_sequence(packets, seq_len):
    sequence = np.zeros(seq_len, dtype=np.float64)

    for packet_index, (packet_time, packet_size) in enumerate(
        sorted(packets, key=lambda packet: packet[0])
    ):
        if packet_index >= seq_len:
            break
        sequence[packet_index] = (
            1.0 if packet_size > 0 else -1.0
        ) * float(packet_time)

    return sequence


def make_open_times(K, arrival_mode, stagger_seconds, rng):
    if arrival_mode == "simultaneous":
        return np.zeros(K, dtype=np.float64)

    if arrival_mode == "fixed":
        return np.arange(K, dtype=np.float64) * float(stagger_seconds)

    if arrival_mode == "uniform":
        return np.sort(
            rng.uniform(0.0, float(stagger_seconds), size=K)
        ).astype(np.float64)

    raise ValueError(f"Unknown arrival mode: {arrival_mode}")


def split_by_direction(cells):
    outgoing = [cell for cell in cells if cell["size"] > 0]
    incoming = [cell for cell in cells if cell["size"] < 0]
    return outgoing, incoming


def mix_direction(
    cells,
    start_time,
    delta_t,
    capacity,
    seed,
    mode,
    padding_per_active_bucket=0,
):
    if not cells:
        return [], [], 0, 0

    if mode == "concurrent":
        packets = [(cell["time"], cell["size"]) for cell in cells]
        return packets, [0.0 for _ in cells], len(cells), 0

    if capacity <= 0:
        raise ValueError("Capacity must be positive for scheduled/full/bounded modes.")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")
    if padding_per_active_bucket < 0:
        raise ValueError("padding_per_active_bucket must be nonnegative.")

    rng = Random(seed)
    cells = sorted(cells, key=lambda cell: (cell["time"], cell["circuit_index"]))

    packets = []
    delays = []
    queue = []
    future_index = 0
    bucket_index = 0
    real_count = 0
    dummy_count = 0

    while future_index < len(cells) or queue:
        bucket_start = start_time + bucket_index * delta_t
        bucket_end = bucket_start + delta_t

        arrivals = []
        while future_index < len(cells) and cells[future_index]["time"] < bucket_end:
            arrivals.append(cells[future_index])
            future_index += 1

        eligible = queue + arrivals
        eligible.sort(
            key=lambda cell: (
                cell["time"],
                cell["circuit_index"],
                rng.random(),
            )
        )

        real_cells = eligible[:capacity]
        queue = eligible[capacity:]

        if mode == "scheduled":
            output_slots = len(real_cells)
        elif mode == "full":
            output_slots = capacity
        elif mode == "bounded":
            output_slots = len(real_cells)
            if real_cells:
                output_slots += min(
                    int(padding_per_active_bucket),
                    capacity - len(real_cells),
                )
        else:
            raise ValueError(f"Unknown mixing mode: {mode}")

        dummy_sign = 1.0
        if real_cells:
            dummy_sign = 1.0 if real_cells[0]["size"] > 0 else -1.0

        for slot in range(output_slots):
            output_time = bucket_start + ((slot + 0.5) * delta_t / float(capacity))

            if slot < len(real_cells):
                cell = real_cells[slot]
                packets.append((output_time, cell["size"]))
                delays.append(max(0.0, output_time - cell["time"]))
                real_count += 1
            else:
                packets.append((output_time, dummy_sign * DATASIZE))
                dummy_count += 1

        bucket_index += 1

    return packets, delays, real_count, dummy_count


def overlap_fraction(open_times, close_times):
    if len(open_times) < 2:
        return 0.0

    start_time = float(np.min(open_times))
    end_time = float(np.max(close_times))
    total_duration = end_time - start_time

    if total_duration <= 0:
        return 0.0

    boundaries = sorted(set(open_times.tolist() + close_times.tolist()))
    overlap_duration = 0.0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue

        midpoint = (left + right) / 2.0
        active_count = np.sum((open_times <= midpoint) & (close_times > midpoint))
        if active_count >= 2:
            overlap_duration += right - left

    return float(overlap_duration / total_duration)


def mix_live_pool(
    traces,
    open_times,
    delta_t,
    N_out,
    N_in,
    seed,
    mode,
    padding_per_active_bucket=0,
):
    all_cells = []
    close_times = []

    for circuit_index, (trace, open_time) in enumerate(zip(traces, open_times)):
        cells = trace_to_live_cells(trace, circuit_index, open_time)
        all_cells.extend(cells)
        close_times.append(
            max(cell["time"] for cell in cells) if cells else float(open_time)
        )

    if not all_cells:
        return [], {
            "close_times": np.asarray(close_times, dtype=np.float64),
            "real_cells_out": 0,
            "dummy_cells": 0,
            "mixed_cells": 0,
            "delay_mean": 0.0,
            "delay_p50": 0.0,
            "delay_p95": 0.0,
            "delay_max": 0.0,
            "mixed_duration": 0.0,
            "completion_delay": 0.0,
        }

    outgoing, incoming = split_by_direction(all_cells)
    scheduler_start = float(np.min(open_times))

    out_packets, out_delays, out_real, out_dummy = mix_direction(
        outgoing,
        scheduler_start,
        delta_t,
        N_out,
        seed + 1,
        mode,
        padding_per_active_bucket,
    )
    in_packets, in_delays, in_real, in_dummy = mix_direction(
        incoming,
        scheduler_start,
        delta_t,
        N_in,
        seed + 2,
        mode,
        padding_per_active_bucket,
    )

    mixed_packets = [
        (packet_time, abs(packet_size)) for packet_time, packet_size in out_packets
    ]
    mixed_packets.extend(
        (packet_time, -abs(packet_size)) for packet_time, packet_size in in_packets
    )
    mixed_packets.sort(key=lambda packet: packet[0])

    delays = np.asarray(out_delays + in_delays, dtype=np.float64)
    close_times = np.asarray(close_times, dtype=np.float64)

    if mixed_packets:
        mixed_times = np.asarray([packet[0] for packet in mixed_packets], dtype=np.float64)
        mixed_duration = float(mixed_times.max() - mixed_times.min())
        completion_delay = max(0.0, float(mixed_times.max() - np.max(close_times)))
    else:
        mixed_duration = 0.0
        completion_delay = 0.0

    return mixed_packets, {
        "close_times": close_times,
        "real_cells_out": int(out_real + in_real),
        "dummy_cells": int(out_dummy + in_dummy),
        "mixed_cells": int(len(mixed_packets)),
        "delay_mean": float(np.mean(delays)) if delays.size else 0.0,
        "delay_p50": float(np.percentile(delays, 50)) if delays.size else 0.0,
        "delay_p95": float(np.percentile(delays, 95)) if delays.size else 0.0,
        "delay_max": float(np.max(delays)) if delays.size else 0.0,
        "mixed_duration": mixed_duration,
        "completion_delay": completion_delay,
    }


def validate_arguments(K, mode, arrival_mode, stagger_seconds, delta_t, N_out, N_in, padding_per_active_bucket, seq_len, progress_every):
    if K < 2:
        raise ValueError("K must be at least 2 for the K>1 study.")
    if mode not in {"concurrent", "scheduled", "full", "bounded"}:
        raise ValueError(f"Unknown mode: {mode}")
    if arrival_mode not in {"simultaneous", "fixed", "uniform"}:
        raise ValueError(f"Unknown arrival mode: {arrival_mode}")
    if stagger_seconds < 0:
        raise ValueError("stagger_seconds must be nonnegative.")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")
    if N_out < 1 or N_in < 1:
        raise ValueError("N_out and N_in must both be at least 1.")
    if padding_per_active_bucket < 0:
        raise ValueError("padding_per_active_bucket must be nonnegative.")
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if progress_every < 1:
        raise ValueError("progress_every must be positive.")


def build_split(
    input_path,
    output_path,
    metadata_path,
    K,
    delta_t,
    N_out,
    N_in,
    mode,
    arrival_mode,
    stagger_seconds,
    num_mixed,
    seq_len,
    seed,
    progress_every,
    padding_per_active_bucket,
):
    validate_arguments(
        K,
        mode,
        arrival_mode,
        stagger_seconds,
        delta_t,
        N_out,
        N_in,
        padding_per_active_bucket,
        seq_len,
        progress_every,
    )

    source_data = np.load(input_path)
    X_source = source_data["X"]
    y_source = source_data["y"]

    if X_source.ndim != 2:
        raise ValueError(f"Expected X with 2 dimensions, got {X_source.shape}")
    if y_source.ndim != 1:
        raise ValueError(
            f"Expected single-label source y with 1 dimension, got {y_source.shape}"
        )
    if X_source.shape[0] != y_source.shape[0]:
        raise ValueError(
            f"X/y sample counts differ: {X_source.shape[0]} versus {y_source.shape[0]}."
        )
    if np.any(y_source < 0):
        raise ValueError("Source labels must be nonnegative integers.")

    source_count, _ = X_source.shape
    available_labels = np.unique(y_source).astype(np.int64)
    num_classes = int(np.max(y_source)) + 1

    if K > source_count:
        raise ValueError(f"K={K} is larger than available source traces={source_count}.")
    if K > len(available_labels):
        raise ValueError(
            f"K={K} is larger than the number of available site labels={len(available_labels)}."
        )

    if num_mixed is None:
        num_mixed = source_count
    if num_mixed <= 0:
        raise ValueError("num_mixed must be positive.")

    label_to_indices = {
        int(label): np.flatnonzero(y_source == label)
        for label in available_labels
    }

    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    X_mixed = np.zeros((num_mixed, seq_len), dtype=np.float64)
    y_multihot = np.zeros((num_mixed, num_classes), dtype=np.uint8)

    groups = np.zeros((num_mixed, K), dtype=np.int64)
    source_indices = np.zeros((num_mixed, K), dtype=np.int64)
    open_times_all = np.zeros((num_mixed, K), dtype=np.float64)
    close_times_all = np.zeros((num_mixed, K), dtype=np.float64)

    overlap_all = np.zeros(num_mixed, dtype=np.float64)
    orig_cells_all = np.zeros(num_mixed, dtype=np.int64)
    mixed_cells_all = np.zeros(num_mixed, dtype=np.int64)
    dummy_cells_all = np.zeros(num_mixed, dtype=np.int64)
    real_cells_all = np.zeros(num_mixed, dtype=np.int64)

    delay_mean_all = np.zeros(num_mixed, dtype=np.float64)
    delay_p50_all = np.zeros(num_mixed, dtype=np.float64)
    delay_p95_all = np.zeros(num_mixed, dtype=np.float64)
    delay_max_all = np.zeros(num_mixed, dtype=np.float64)

    source_session_duration_all = np.zeros(num_mixed, dtype=np.float64)
    mixed_duration_all = np.zeros(num_mixed, dtype=np.float64)
    completion_delay_all = np.zeros(num_mixed, dtype=np.float64)
    bandwidth_overhead_all = np.zeros(num_mixed, dtype=np.float64)
    scheduler_latency_overhead_all = np.zeros(num_mixed, dtype=np.float64)

    start_wall_time = time.time()

    print("Building K>1 live mixed split")
    print("-----------------------------")
    print(f"Input source file : {input_path}")
    print(f"Output file       : {output_path}")
    print(f"Mode              : {mode}")
    print(f"K                 : {K}")
    print(f"Arrival mode      : {arrival_mode}")
    print(f"Stagger seconds   : {stagger_seconds}")
    print(f"delta_t           : {delta_t}")
    print(f"N_out / N_in      : {N_out} / {N_in}")
    print(f"Padding/active bucket: {padding_per_active_bucket}")
    print(f"Source traces     : {source_count}")
    print(f"Mixed sessions    : {num_mixed}")
    print(f"Output seq_len    : {seq_len}")

    for mixed_index in range(num_mixed):
        rng = np.random.default_rng(seed + mixed_index)

        selected_labels = rng.choice(
            available_labels,
            size=K,
            replace=False,
        ).astype(np.int64)

        selected_indices = np.asarray(
            [
                rng.choice(label_to_indices[int(label)])
                for label in selected_labels
            ],
            dtype=np.int64,
        )
        selected_traces = [X_source[index] for index in selected_indices]

        open_times = make_open_times(K, arrival_mode, stagger_seconds, rng)
        mixed_packets, meta = mix_live_pool(
            traces=selected_traces,
            open_times=open_times,
            delta_t=delta_t,
            N_out=N_out,
            N_in=N_in,
            seed=seed + mixed_index,
            mode=mode,
            padding_per_active_bucket=padding_per_active_bucket,
        )

        X_mixed[mixed_index] = packets_to_cw_sequence(mixed_packets, seq_len)
        y_multihot[mixed_index, selected_labels] = 1
        groups[mixed_index] = selected_labels
        source_indices[mixed_index] = selected_indices
        open_times_all[mixed_index] = open_times
        close_times_all[mixed_index] = meta["close_times"]

        original_cells = sum(int(np.count_nonzero(trace)) for trace in selected_traces)
        source_start = float(np.min(open_times))
        source_end = float(np.max(meta["close_times"]))
        source_session_duration = source_end - source_start
        mixed_duration = float(meta["mixed_duration"])

        overlap_all[mixed_index] = overlap_fraction(open_times, meta["close_times"])
        orig_cells_all[mixed_index] = original_cells
        mixed_cells_all[mixed_index] = meta["mixed_cells"]
        dummy_cells_all[mixed_index] = meta["dummy_cells"]
        real_cells_all[mixed_index] = meta["real_cells_out"]

        delay_mean_all[mixed_index] = meta["delay_mean"]
        delay_p50_all[mixed_index] = meta["delay_p50"]
        delay_p95_all[mixed_index] = meta["delay_p95"]
        delay_max_all[mixed_index] = meta["delay_max"]

        source_session_duration_all[mixed_index] = source_session_duration
        mixed_duration_all[mixed_index] = mixed_duration
        completion_delay_all[mixed_index] = meta["completion_delay"]
        bandwidth_overhead_all[mixed_index] = (
            float(meta["mixed_cells"] / original_cells - 1.0)
            if original_cells > 0
            else 0.0
        )
        scheduler_latency_overhead_all[mixed_index] = (
            float(mixed_duration / source_session_duration - 1.0)
            if source_session_duration > 0
            else 0.0
        )

        completed = mixed_index + 1
        if completed % progress_every == 0 or completed == num_mixed:
            elapsed = time.time() - start_wall_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (num_mixed - completed) / rate if rate > 0 else float("inf")
            print(
                f"Generated {completed}/{num_mixed} | "
                f"elapsed={elapsed / 60:.1f} min | "
                f"ETA={eta / 60:.1f} min | "
                f"mean overlap={np.mean(overlap_all[:completed]):.4f} | "
                f"mean BW={np.mean(bandwidth_overhead_all[:completed]):.4f} | "
                f"mean completion delay={np.mean(completion_delay_all[:completed]):.4f}s"
            )

    np.savez_compressed(output_path, X=X_mixed, y=y_multihot)
    np.savez_compressed(
        metadata_path,
        groups=groups,
        source_indices=source_indices,
        open_times=open_times_all,
        close_times=close_times_all,
        stagger_delays=open_times_all - open_times_all[:, :1],
        overlap_fraction=overlap_all,
        orig_cells=orig_cells_all,
        mixed_cells=mixed_cells_all,
        dummy_cells=dummy_cells_all,
        real_cells_out=real_cells_all,
        delay_mean=delay_mean_all,
        delay_p50=delay_p50_all,
        delay_p95=delay_p95_all,
        delay_max=delay_max_all,
        source_session_duration=source_session_duration_all,
        mixed_duration=mixed_duration_all,
        completion_delay=completion_delay_all,
        bw_overhead=bandwidth_overhead_all,
        scheduler_lat_overhead=scheduler_latency_overhead_all,
    )

    print(f"Saved attacker dataset: {output_path}")
    print(f"Saved private metadata: {metadata_path}")
    print(f"Mean overlap fraction: {np.mean(overlap_all):.6f}")
    print(f"Mean bandwidth overhead: {np.mean(bandwidth_overhead_all):.6f}")
    print(
        "Mean scheduler latency overhead: "
        f"{np.mean(scheduler_latency_overhead_all):.6f}"
    )
    print(f"Mean completion delay (seconds): {np.mean(completion_delay_all):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a live K>1 cross-circuit dataset from one source split."
    )

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["concurrent", "scheduled", "full", "bounded"],
        default="bounded",
    )
    parser.add_argument(
        "--padding_per_active_bucket",
        type=int,
        default=1,
        help="Maximum dummy cells added in an active bucket; used only by bounded mode.",
    )
    parser.add_argument(
        "--arrival_mode",
        choices=["simultaneous", "fixed", "uniform"],
        default="fixed",
    )
    parser.add_argument("--stagger_seconds", type=float, default=5.0)
    parser.add_argument("--delta_t", type=float, default=0.01)
    parser.add_argument("--N_out", type=int, default=1)
    parser.add_argument("--N_in", type=int, default=1)
    parser.add_argument(
        "--num_mixed",
        type=int,
        default=None,
        help="Default: generate one mixed session per source trace.",
    )
    parser.add_argument("--seq_len", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--progress_every", type=int, default=100)

    args = parser.parse_args()

    build_split(
        input_path=args.input_path,
        output_path=args.output_path,
        metadata_path=args.metadata_path,
        K=args.K,
        delta_t=args.delta_t,
        N_out=args.N_out,
        N_in=args.N_in,
        mode=args.mode,
        arrival_mode=args.arrival_mode,
        stagger_seconds=args.stagger_seconds,
        num_mixed=args.num_mixed,
        seq_len=args.seq_len,
        seed=args.seed,
        progress_every=args.progress_every,
        padding_per_active_bucket=args.padding_per_active_bucket,
    )


if __name__ == "__main__":
    main()
