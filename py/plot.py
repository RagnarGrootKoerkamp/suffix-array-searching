import sa_layout
import matplotlib.pyplot as plt

# placeholder for now to see some results
START_POW2 = 10
STOP_POW2 = 20
NUM_REPEATS = 20

def plot_results(sizes, timings, comparisons):
    plt.plot(sizes, timings)
    plt.xlabel('Index')          # Label for the x-axis
    plt.ylabel('Values')         # Label for the y-axis
    plt.title('Simple Line Plot') # Title of the plot
    plt.legend()                 # Show legend
    plt.grid(True)               # Show grid for readability
    plt.show()

b = sa_layout.BenchmarkSortedArray(2**20)
sizes = [2**x for x in range(1, 20)]
timings, comparisons = b.benchmark("basic_binsearch", 1, 20, 20)
print(timings, comparisons)
plot_results(sizes, timings, comparisons)