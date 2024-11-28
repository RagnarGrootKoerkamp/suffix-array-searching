#!/usr/bin/python
import sa_layout
import numpy as np
import matplotlib.pyplot as plt

# placeholder for now to see some results
START_POW2 = 10
STOP_POW2 = 27
NUM_REPEATS = 1000000
to_try = ["eytzinger_prefetched", "btree_basic_16", "btree_branchless_16"]

def plot_results(sizes, names, timings, comparisons, filename="plot.pdf"):
    fig, ax = plt.subplots()
    for name, timing in zip(names, timings):
        ax.plot(4 * np.array(sizes), timing, label=name)
    # Customize the plot
    ax.set_title("Performance plot")
    ax.set_xlabel("size of inputs")
    ax.set_ylabel("time in ns")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True)  # Show grid for readability
    fig.savefig(filename, bbox_inches="tight")


b = sa_layout.BenchmarkSortedArray()
sizes = [2**x for x in range(START_POW2, STOP_POW2)]
timings = []
comparisons = []
print("START BENCHING")
for name in to_try:
    timing, comparison = b.benchmark(name, START_POW2, STOP_POW2, NUM_REPEATS)
    timings.append(timing)
    comparisons.append(comparison)

plot_results(sizes, to_try, timings, comparisons, filename="plot.png")
