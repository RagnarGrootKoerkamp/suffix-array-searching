#!/usr/bin/python
import sa_layout
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import plot_results

if __name__ == "__main__":
    b = sa_layout.BenchmarkSortedArray()

    # placeholder for now to see some results
    START_POW2 = 5
    STOP_POW2 = 28
    NUM_QUERIES = 1000000

    print("Starting benchmarks..")
    results = b.benchmark(START_POW2, STOP_POW2, NUM_QUERIES)

    plot_results(results, "plot.svg")
