import os
import argparse
import numpy as np

from Tamaraw import Anoa, AnoaPad, set_parameters, get_parameters

IN_PATH = "datasets/CW.npz"
DATASIZE = 800  # must match Tamaraw.py


def format_float_for_name(x):
    """
    Produce a filesystem-friendly float string for dataset names.
    Examples:
      0.005 -> "0p005"
      0.01  -> "0p01"
      0.02  -> "0p02"
      1.0   -> "1"
    """
    s = "{:.10g}".format(float(x))
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    return s


def build_output_path(padl, p_in, p_out, l_value, g_value):
    p_in_s = format_float_for_name(p_in)
    p_out_s = format_float_for_name(p_out)
    return (
        f"datasets/CW_tamaraw_padl{padl}"
        f"_pin{p_in_s}"
        f"_pout{p_out_s}"
        f"_L{l_value}"
        f"_G{g_value}.npz"
    )


def cw_trace_to_packets(trace):
    """
    Convert one CW trace into Tamaraw packet format.

    Input:
      trace: 1D array where each value = direction * timestamp

    Output:
      list of [time, size]
    """
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


def packets_to_cw_sequence(packets, seq_len=10000):
    """
    Convert defended packet list back to CW-style sequence.

    Output sequence:
      seq[i] = direction * timestamp
    """
    packets = sorted(packets, key=lambda x: x[0])
    seq = np.zeros(seq_len, dtype=np.float64)

    for i, (t, sz) in enumerate(packets):
        if i >= seq_len:
            break
        direction = 1.0 if sz > 0 else -1.0
        seq[i] = direction * t

    return seq


def defend_one_trace(trace, padl):
    """
    Defend a single CW trace and return:
      defended_sequence, original_packet_count, defended_packet_count
    """
    packets = cw_trace_to_packets(trace)
    total_orig = len(packets)

    list1 = [pkt[:] for pkt in packets]
    list2 = []
    params = [""]

    Anoa(list1, list2, params)
    list2 = sorted(list2, key=lambda x: x[0])

    list3 = []
    AnoaPad(list2, list3, padL=padl, method=0)
    total_def = len(list3)

    defended_seq = packets_to_cw_sequence(list3, seq_len=len(trace))
    return defended_seq, total_orig, total_def


def defend_all_traces(padl=100, p_in=0.012, p_out=0.04, l_value=0, g_value=0):
    d = np.load(IN_PATH)
    X = d["X"]
    y = d["y"]

    N, T = X.shape
    X_def = np.zeros_like(X, dtype=np.float64)

    set_parameters(p_in=p_in, p_out=p_out, l=l_value, g=g_value)
    active_params = get_parameters()

    total_orig = 0
    total_def = 0

    print("Active Tamaraw parameters:")
    print(f"  padL = {padl}")
    print(f"  p_in = {active_params['p_in']}")
    print(f"  p_out = {active_params['p_out']}")
    print(f"  L = {active_params['L']}")
    print(f"  G = {active_params['G']}")

    for i in range(N):
        trace = X[i]
        defended_seq, orig_count, def_count = defend_one_trace(trace, padl=padl)

        X_def[i] = defended_seq
        total_orig += orig_count
        total_def += def_count

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{N} traces")

    bw_overhead = total_def / total_orig - 1.0
    print(f"Average bandwidth overhead: {bw_overhead * 100:.2f}%")

    out_path = build_output_path(
        padl=padl,
        p_in=p_in,
        p_out=p_out,
        l_value=l_value,
        g_value=g_value,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(out_path, X=X_def, y=y)
    print(f"Saved defended dataset to: {out_path}")

    return out_path, bw_overhead


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padl", type=int, default=100)
    parser.add_argument("--p_in", type=float, default=0.012)
    parser.add_argument("--p_out", type=float, default=0.04)
    parser.add_argument("--L", type=int, default=0)
    parser.add_argument("--G", type=int, default=0)

    args = parser.parse_args()

    defend_all_traces(
        padl=args.padl,
        p_in=args.p_in,
        p_out=args.p_out,
        l_value=args.L,
        g_value=args.G,
    )
