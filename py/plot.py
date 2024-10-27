import sa_layout
import matplotlib.pyplot as plt

# placeholder for now to see some results
START_POW2 = 10
STOP_POW2 = 28
NUM_REPEATS = 50
to_try = ["basic_binsearch", "basic_binsearch_branchless"]

def plot_results(sizes, timings, comparisons, name="plot.pdf"):
    fig, ax = plt.subplots()
    for timing in timings:
        ax.plot(sizes, timing)
    # Customize the plot
    ax.set_title("Performance plot")
    ax.set_xlabel("size of inputs")
    ax.set_ylabel("time in ns")
    ax.set_xscale('log', base=2)
    ax.grid(True)               # Show grid for readability
    fig.savefig(name, bbox_inches="tight")

b = sa_layout.BenchmarkSortedArray(2**STOP_POW2)
sizes = [2**x for x in range(START_POW2, STOP_POW2)]
timings = []
comparisons = []
for name in to_try:
    timing, comparison = b.benchmark(name, START_POW2, STOP_POW2, NUM_REPEATS)
    timings.append(timing)
    comparisons.append(comparison)

plot_results(sizes, timings, comparisons, name="basic_binsearch")