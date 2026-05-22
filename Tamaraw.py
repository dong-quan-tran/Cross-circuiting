# Tamaraw.py
# Anoa/Tamaraw-style defense with tuning-friendly parameters.
#
# Current behavior:
# - Packet schedule controlled by P_OUT (outgoing interval) and P_IN (incoming interval).
# - End padding controlled by padL in AnoaPad().
# - L and G are accepted and stored for future tuning, but are not yet active in the logic.
#
# Default parameters preserve the original behavior:
#   P_OUT = 0.04
#   P_IN  = 0.012
#   L = 0
#   G = 0

import math
import random

DATASIZE = 800

# Kept from original file for compatibility / future use
tardist = [[], []]
defpackets = []

# Global tuning parameters (defaults preserve original behavior)
P_OUT = 0.04   # outgoing packet interval
P_IN = 0.012   # incoming packet interval
L = 0          # reserved for future tuning
G = 0          # reserved for future tuning


def set_parameters(pin=None, pout=None, l=None, g=None):
    """
    Update global Tamaraw parameters in a tuning-friendly way.

    Parameters
    ----------
    pin : float or None
        Incoming packet interval.
    pout : float or None
        Outgoing packet interval.
    l : int or float or None
        Reserved parameter for future Tamaraw tuning.
    g : int or float or None
        Reserved parameter for future Tamaraw tuning.
    """
    global P_IN, P_OUT, L, G

    if pin is not None:
        P_IN = float(pin)
    if pout is not None:
        P_OUT = float(pout)
    if l is not None:
        L = l
    if g is not None:
        G = g


def get_parameters():
    """
    Return the currently active Tamaraw parameters.
    """
    return {
        "pin": P_IN,
        "pout": P_OUT,
        "L": L,
        "G": G,
    }


def fsign(num):
    if num > 0:
        return 0
    else:
        return 1


def rsign(num):
    if num == 0:
        return 1
    else:
        return abs(num) / num


def AnoaTime(parameters):
    """
    Return the packet interval for the requested direction.

    parameters[0] = direction
        0 -> outgoing
        1 -> incoming

    parameters[1] = method
        currently only method 0 is supported

    Extra elements in parameters are ignored for compatibility.
    """
    direction = parameters[0]  # 0 out, 1 in
    method = parameters[1]

    if method == 0:
        if direction == 0:
            return P_OUT
        if direction == 1:
            return P_IN

    raise ValueError("Unsupported AnoaTime method: {}".format(method))


def AnoaPad(list1, list2, padL, method):
    """
    Pad the defended packet sequence up to a randomized multiple of padL.

    Parameters
    ----------
    list1 : list
        Input packet list (already defended by Anoa), format [[time, size], ...]
    list2 : list
        Output list that receives the padded trace
    padL : int
        Padding block size
    method : int
        Currently only method 0 is supported
    """
    lengths = [0, 0]
    times = [0, 0]

    for x in list1:
        if x[1] > 0:
            lengths[0] += 1
            times[0] = x[0]
        else:
            lengths[1] += 1
            times[1] = x[0]
        list2.append(x)

    for j in range(0, 2):
        curtime = times[j]
        topad = -int(math.log(random.uniform(0.00001, 1), 2) - 1)

        if method == 0:
            topad = (lengths[j] / padL + topad) * padL

        while lengths[j] < topad:
            curtime += AnoaTime([j, 0])
            if j == 0:
                list2.append([curtime, DATASIZE])
            else:
                list2.append([curtime, -DATASIZE])
            lengths[j] += 1


def Anoa(list1, list2, parameters):
    """
    Defend a trace by sending packets at a constant packet rate.

    Parameters
    ----------
    list1 : list
        Input packet list [[time, size], ...]
        NOTE: list1 may be modified in-place, matching original behavior.
    list2 : list
        Output packet list
    parameters : list
        Mutable list used by the original code to store description text.
        We preserve that interface for compatibility.
    """
    if not list1:
        return

    starttime = list1[0][0]
    times = [starttime, starttime]
    curtime = starttime
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

    if method == 1:
        parameters[0] = "Time-split varying bandwidth, split by 0.1 seconds. "
        parameters[0] += "Tolerance: 2x."

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
    import os
    import sys

    args = sys.argv[1:]

    # Optional CLI compatibility for standalone use:
    #   python Tamaraw.py pin pout L G
    # If fewer than 4 values are provided, defaults are used for missing ones.
    if len(args) >= 1:
        P_IN = float(args[0])
    if len(args) >= 2:
        P_OUT = float(args[1])
    if len(args) >= 3:
        L = int(float(args[2]))
    if len(args) >= 4:
        G = int(float(args[3]))

    sitenum = 100
    instnum = 90
    fold = "batch/"
    foldout = "batch-tamaraw/"

    if not os.path.exists(foldout):
        os.makedirs(foldout)

    for site in range(0, sitenum):
        print(site)
        for inst in range(0, instnum):
            packets = []
            with open(fold + str(site) + "-" + str(inst), "r") as f:
                lines = f.readlines()
                starttime = float(lines[0].split("\t")[0])
                for x in lines:
                    x = x.split("\t")
                    packets.append([float(x[0]) - starttime, int(x[1])])

            list2 = []
            parameters = [""]

            Anoa(packets, list2, parameters)
            list2 = sorted(list2, key=lambda x: x[0])

            list3 = []
            AnoaPad(list2, list3, 100, 0)

            with open(foldout + str(site) + "-" + str(inst), "w") as fout:
                for x in list3:
                    fout.write(str(x[0]) + "\t" + str(x[1]) + "\n")
