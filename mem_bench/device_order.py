#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams["axes.facecolor"] = "white"


device_color_palette = [
    "#378ABD",
    "#FFB33A",
    "#7EC75B",
    "#DA5252",
    "#793B67",
    "#10CFCC",
    "#FFE100",
    "#09047f",
    "#296F20",
]

order = [
    "a40",
    "l40",
    "v100",
    "a100",
    "gh200",
    "mi100",
    "mi210",
    "mi300x",
    "rx6900xt",
    "mi300a",
    "a100_40",
    "h100_pcie",
]


long_order = [
    "NVIDIA A40",
    "NVIDIA L40",
    "Tesla V100",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA GH200 480GB",
    "AMD Instinct MI100",
    "AMD Instinct MI210",
    "AMD Instinct MI300X",
    "AMD Radeon RX 6900 XT",
    "NVIDIA A100-SXM4-40G",
]


def getOrderNumber(f, use_order=order):
    for o in range(len(use_order)):
        if f.startswith(use_order[o]):
            return o
    return len(use_order) + 1


def getDeviceColor(f):
    n = getOrderNumber(f)
    if n >= len(device_color_palette):
        n = getOrderNumber(f, use_order=long_order)
    if n >= len(device_color_palette):
        return "C" + str(n - len(device_color_palette))

    return device_color_palette[n]


lineStyle = {"linewidth": 2.0, "alpha": 1, "markersize": 3, "marker": ""}
