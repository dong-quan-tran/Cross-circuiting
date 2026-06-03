import os
import json
import argparse
import numpy as np

from Tamaraw import Anoa, AnoaPad, set_parameters, get_parameters

IN_PATH = "datasets/CW.npz"
DATASIZE = 800


def format_float_for_name(x):
    s = "{:.10g}".format(float(x))
    s = s.replace("-", "m").replace(".", "p")
    return s


def build_output_path(legacy_padl, p_in, p_out, l_value, g_value):
    p_in_s = format_float_for_name(p_in)
    p_out_s = format_float_for_name(p_out)
    return (
        f"datasets/CW_tamaraw_legacyPadl{legacy_padl}"
        f"_pin{p_in_s}"
        f"_pout{p_out_s}"
        f"_L{l_value}"
        f"_G{g_value}.npz"
    )


def build_metrics_path(npz_path):
    return npz_path.replace(".npz", "_metrics.json")


def cw_trace_to_packets(trace):
    packets = []
    for v in trace:
        if v == 0:
            continue
        real_time = abs(v)
        size = DATASIZE if v > 0 else -DATASIZE
        packets.append([real_time, size])

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
        seq[i] = direction * t

    return seq


def trace_duration(packets):
    if not packets:
        return 0.0
    return max(t for t, _ in packets)


def defend_one_trace(trace, legacy_padl):
    packets = cw_trace_to_packets(trace)
    total_orig_packets = len(packets)
    orig_duration = trace_duration(packets)

    list1 = [pkt[:] for pkt in packets]
    list2 = []
    params = [""]

    Anoa(list1, list2, params)
    list2 = sorted(list2, key=lambda x: x[0])

    list3 = []
    AnoaPad(list2, list3, padL=legacy_padl, method=0)
    list3 = sorted(list3, key=lambda x: x[0])

    total_def_packets = len(list3)
    def_duration = trace_duration(list3)

    defended_seq = packets_to_cw_sequence(list3, seq_len=len(trace))

    return {
        "defended_seq": defended_seq,
        "orig_packets": total_orig_packets,
        "def_packets": total_def_packets,
        "orig_duration": orig_duration,
        "def_duration": def_duration,
    }


def defend_all_traces(legacy_padl=100, p_in=0.012, p_out=0.04, l_value=0, g_value=0):
    d = np.load(IN_PATH)
    X = d["X"]
    y = d["y"]

    N, T = X.shape
    X_def = np.zeros_like(X, dtype=np.float64)

    set_parameters(p_in=p_in, p_out=p_out, l=l_value, g=g_value)
    active = get_parameters()

    total_orig_packets = 0
    total_def_packets = 0
    total_orig_duration = 0.0
    total_def_duration = 0.0

    if int(active["L"]) > 0:
        effective_ell = 500 * int(active["L"])
        mode = "WANG"
        padl_note = f"legacy_padl={legacy_padl} ignored because L>0"
    else:
        effective_ell = int(legacy_padl)
        mode = "LEGACY"
        padl_note = f"legacy_padl={legacy_padl} used because L<=0"

    print("Active Tamaraw parameters")
    print("-------------------------")
    print(f"mode                : {mode}")
    print(f"p_in                : {active['p_in']}")
    print(f"p_out               : {active['p_out']}")
    print(f"L                   : {active['L']}")
    print(f"G                   : {active['G']}")
    print(f"effective ell       : {effective_ell}")
    print(f"legacy padL note    : {padl_note}")

    for i in range(N):
        result = defend_one_trace(X[i], legacy_padl=legacy_padl)
        X_def[i] = result["defended_seq"]

        total_orig_packets += result["orig_packets"]
        total_def_packets += result["def_packets"]
        total_orig_duration += result["orig_duration"]
        total_def_duration += result["def_duration"]

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{N} traces")

    bandwidth_overhead = total_def_packets / total_orig_packets - 1.0
    avg_orig_duration = total_orig_duration / N
    avg_def_duration = total_def_duration / N

    if avg_orig_duration > 0:
        latency_overhead = avg_def_duration / avg_orig_duration - 1.0
    else:
        latency_overhead = 0.0

    print(f"Average bandwidth overhead: {bandwidth_overhead * 100:.2f}%")
    print(f"Average original duration : {avg_orig_duration:.6f}s")
    print(f"Average defended duration : {avg_def_duration:.6f}s")
    print(f"Average latency overhead  : {latency_overhead * 100:.2f}%")

    out_path = build_output_path(
        legacy_padl=legacy_padl,
        p_in=p_in,
        p_out=p_out,
        l_value=l_value,
        g_value=g_value,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, X=X_def, y=y)
    print(f"Saved defended dataset to: {out_path}")

    metrics = {
        "input_path": IN_PATH,
        "output_path": out_path,
        "mode": mode,
        "legacy_padl": int(legacy_padl),
        "p_in": float(active["p_in"]),
        "p_out": float(active["p_out"]),
        "L": int(active["L"]),
        "G": int(active["G"]),
        "effective_ell": int(effective_ell),
        "num_traces": int(N),
        "total_orig_packets": int(total_orig_packets),
        "total_def_packets": int(total_def_packets),
        "bandwidth_overhead_fraction": float(bandwidth_overhead),
        "bandwidth_overhead_percent": float(bandwidth_overhead * 100.0),
        "avg_orig_duration_sec": float(avg_orig_duration),
        "avg_def_duration_sec": float(avg_def_duration),
        "latency_overhead_fraction": float(latency_overhead),
        "latency_overhead_percent": float(latency_overhead * 100.0),
    }

    metrics_path = build_metrics_path(out_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")

    return out_path, metrics_path, bandwidth_overhead, latency_overhead


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--legacy_padl",
        type=int,
        default=100,
        help="Legacy fallback pad length. Used only when L <= 0."
    )
    parser.add_argument(
        "--p_in",
        type=float,
        default=0.012,
        help="Incoming packet interval in seconds."
    )
    parser.add_argument(
        "--p_out",
        type=float,
        default=0.04,
        help="Outgoing packet interval in seconds."
    )
    parser.add_argument(
        "--L",
        type=int,
        default=0,
        help="Wang mode length multiplier. If L > 0, ell = 500 * L."
    )
    parser.add_argument(
        "--G",
        type=int,
        default=0,
        help="Wang mode geometric parameter. If G <= 0, treated as 1."
    )

    args = parser.parse_args()

    defend_all_traces(
        legacy_padl=args.legacy_padl,
        p_in=args.p_in,
        p_out=args.p_out,
        l_value=args.L,
        g_value=args.G,
    )
