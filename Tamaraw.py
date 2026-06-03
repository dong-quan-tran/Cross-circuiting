# Tamaraw.py
# Wang-inspired Tamaraw end-padding:
#   - If L > 0, sequence-end padding uses ell = 500 * L
#   - If G > 0, extra blocks are sampled geometrically in Wang's form
#   - legacy/custom padL is used only when L <= 0
#
# Wang Section 5.1:
#   If A is the smallest multiple of ell >= current length, final target is:
#       A + k * ell
#   where k is drawn from a geometric distribution controlled by G.
#
# In previous experiments in the paper:
#   L = 1, G = 1

import random

DATASIZE = 800

tardist = [[], []]
defpackets = []

P_OUT = 0.04
P_IN = 0.012
L = 0
G = 0


def set_parameters(p_in=None, p_out=None, l=None, g=None):
    global P_IN, P_OUT, L, G

    if p_in is not None:
        P_IN = float(p_in)
    if p_out is not None:
        P_OUT = float(p_out)
    if l is not None:
        L = int(float(l))
    if g is not None:
        G = int(float(g))


def get_parameters():
    return {
        "p_in": P_IN,
        "p_out": P_OUT,
        "L": L,
        "G": G,
    }


def fsign(num):
    return 0 if num > 0 else 1


def rsign(num):
    if num == 0:
        return 1
    return abs(num) / num


def AnoaTime(parameters):
    direction = parameters[0]
    method = parameters[1]

    if method == 0:
        if direction == 0:
            return P_OUT
        if direction == 1:
            return P_IN

    raise ValueError("Unsupported AnoaTime method: {}".format(method))


def _ceil_to_multiple(x, m):
    x = int(x)
    m = int(m)
    return ((x + m - 1) // m) * m


def _effective_ell(padL):
    if int(L) > 0:
        return 500 * int(L)

    if padL is None or int(padL) <= 0:
        raise ValueError("padL must be positive when L <= 0")

    return int(padL)


def _sample_wang_geometric_blocks(g_value):
    """
    Sample k from:
      Pr[X = k] = (1 - 1/G)^k * (1/G),  k = 0,1,2,...

    Robustness convention:
      G <= 1 => k = 0
    """
    g_value = int(g_value)

    if g_value <= 1:
        return 0

    p = 1.0 / float(g_value)
    k = 0
    while random.random() >= p:
        k += 1
    return k


def AnoaPad(list1, list2, padL, method):
    if method != 0:
        raise ValueError("Unsupported AnoaPad method: {}".format(method))

    lengths = [0, 0]
    times = [0.0, 0.0]

    for x in list1:
        if x[1] > 0:
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)

    ell = _effective_ell(padL)
    g_eff = int(G) if int(G) > 0 else 1

    for j in range(2):
        curtime = times[j]
        base_target = _ceil_to_multiple(lengths[j], ell)
        extra_blocks = _sample_wang_geometric_blocks(g_eff)
        final_target = base_target + extra_blocks * ell

        while lengths[j] < final_target:
            curtime += AnoaTime([j, 0])
            if j == 0:
                list2.append([curtime, DATASIZE])
            else:
                list2.append([curtime, -DATASIZE])
            lengths[j] += 1


def Anoa(list1, list2, parameters):
    if not list1:
        return

    starttime = list1[0][0]
    times = [starttime, starttime]
    lengths = [0, 0]
    datasize = DATASIZE
    method = 0

    if method == 0:
        parameters[0] = (
            "Constant packet rate: "
            + str(AnoaTime([0, 0]))
            + ", "
            + str(AnoaTime([1, 0]))
            + ". "
        )
        parameters[0] += "Data size: " + str(datasize) + ". "
        parameters[0] += "L: " + str(L) + ". "
        parameters[0] += "G: " + str(G) + ". "

    listind = 0

    while listind < len(list1):
        if (
            times[0] + AnoaTime([0, method, times[0] - starttime])
            < times[1] + AnoaTime([1, method, times[1] - starttime])
        ):
            cursign = 0
        else:
            cursign = 1

        times[cursign] += AnoaTime([cursign, method, times[cursign] - starttime])
        curtime = times[cursign]

        tosend = datasize
        while (
            listind < len(list1)
            and list1[listind][0] <= curtime
            and fsign(list1[listind][1]) == cursign
            and tosend > 0
        ):
            if tosend >= abs(list1[listind][1]):
                tosend -= abs(list1[listind][1])
                listind += 1
            else:
                list1[listind][1] = (
                    abs(list1[listind][1]) - tosend
                ) * rsign(list1[listind][1])
                tosend = 0

        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])

        lengths[cursign] += 1


if __name__ == "__main__":
    print("Tamaraw module loaded. Use from run_Tamaraw_CW.py.")
