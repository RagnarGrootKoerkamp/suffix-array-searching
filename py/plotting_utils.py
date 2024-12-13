from matplotlib import pyplot as plt
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
    plt.close("all")
    fig, ax = plt.subplots()
    for name, rs in sorted(results.items()):
        ls = "solid"
        if "ptr" in name:
            ls = "dashed"
        ax.plot([r[0] for r in rs], [r[1] for r in rs], label=name, ls=ls)

    # Customize the plot
    ax.set_xlabel("Array size (bytes)")
    ax.set_ylabel("Time per query (ns)")
    ax.set_xscale("log", base=2)
    secax = ax.secondary_xaxis("top", functions=(lambda x: x / 4, lambda x: x * 4))
    secax.set_xscale("log", base=2)
    secax.set_xlabel("Array length (u32)")

    for L in caches():
        ax.axvline(x=L, linestyle="--", color="blue")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.grid(which="minor", color="gray", alpha=0.2)
    fig.savefig(out, bbox_inches="tight")
    plt.show()
