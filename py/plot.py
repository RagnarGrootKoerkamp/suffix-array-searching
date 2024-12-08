#!/usr/bin/python
import sa_layout
import numpy as np
import matplotlib.pyplot as plt

# placeholder for now to see some results
START_POW2 = 5
STOP_POW2 = 28

L1 = 32768
L2 = 524288
L3 = 4194304

NUM_REPEATS = 1000000
to_try = ["basic_binsearch", "basic_binsearch_branchless", "eytzinger", "eytzinger_prefetched", "btree_basic_16", "btree_branchless_16", "btree_simd_16"]

# def plot_results(sizes, names, timings, comparisons, filename="plot.pdf"):
#     fig, ax = plt.subplots()
#     for name, timing in zip(names, timings):
#         ax.plot(4 * np.array(sizes), timing, label=name)
#     # Customize the plot
#     ax.set_title("Performance plot")
#     ax.set_xlabel("size of inputs")
#     ax.set_ylabel("time in ns")
#     ax.set_xscale("log", base=2)
#     ax.legend()
#     ax.grid(True)  # Show grid for readability
#     fig.savefig(filename, bbox_inches="tight")

def plot_results(benchmark, names, filename):
    timings = [benchmark[name][0] for name in names]
    comparisons = [benchmark[name][1] for name in names]
    fig, ax = plt.subplots()
    for name, timing in zip(names, timings):
        ax.plot(4 * np.array(sizes), timing, label=name)
    # Customize the plot
    ax.set_title("Performance plot")
    ax.set_xlabel("size of inputs")
    ax.set_ylabel("time in ns")
    ax.set_xscale("log", base=2)
    ax.axvline(x=L1, linestyle="--", color="blue")
    ax.axvline(x=L2, linestyle="--", color="blue")
    ax.axvline(x=L3, linestyle="--", color="blue")
    ax.legend()
    ax.grid(True)  # Show grid for readability
    fig.savefig(filename, bbox_inches="tight")

b = sa_layout.BenchmarkSortedArray()
sizes = [2**x for x in range(START_POW2, STOP_POW2)]
timings = []
comparisons = []
print("START BENCHING")

for name in to_try:
    b.add_func_to_bm(name)

benchmarks = b.benchmark(START_POW2, STOP_POW2, NUM_REPEATS)

binsearches = ["basic_binsearch", "basic_binsearch_branchless", "eytzinger", "eytzinger_prefetched"]
btrees_plus_eyetzinger = ["eytzinger", "eytzinger_prefetched", "btree_basic_16", "btree_simd_16"]
plot_results(benchmarks, binsearches, "plot_binsearches.png")
plot_results(benchmarks, btrees_plus_eyetzinger, "plot_btrees.png")
