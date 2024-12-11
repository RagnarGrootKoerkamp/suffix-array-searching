#!/usr/bin/python
import sa_layout
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def caches():
    sizes = []
    # Note: index1 for me is the L1 instruction cache.
    # Note: All read strings are eg 32K.
    for i in [0, 2, 3]:
        t = Path(f"/sys/devices/system/cpu/cpu0/cache/index{i}/size").read_text()
        sizes.append(int(t[:-2]) * 1024)
    return sizes


def plot_results(results, out):
    fig, ax = plt.subplots()
    print(results)
    for name, rs in results.items():
        ax.plot([r[0] for r in rs], [r[1] for r in rs], label=name)
    # Customize the plot
    ax.set_title("Performance plot")
    ax.set_xlabel("size of inputs")
    ax.set_ylabel("time in ns")
    ax.set_xscale("log", base=2)
    for L in caches():
        ax.axvline(x=L, linestyle="--", color="blue")
    ax.legend()
    ax.grid(True)  # Show grid for readability
    fig.savefig(out, bbox_inches="tight")
    fig.show()


b = sa_layout.BenchmarkSortedArray()

# placeholder for now to see some results
START_POW2 = 5
STOP_POW2 = 28
NUM_QUERIES = 1000000

results = b.benchmark(START_POW2, STOP_POW2, NUM_QUERIES)

plot_results(results, "plot.svg")
