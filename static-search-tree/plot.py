#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tabulate
from matplotlib.ticker import LogLocator

l1_size = 192 * 1024 / 6
l2_size = 1.5 * 1024 * 1024 / 6
l3_size = 12 * 1024 * 1024

palette = None
dashes = {"": "", "Latency": (1, 1), "Prefetch": (2, 1)}


def plot(experiment_name, title, data, names, skip=0, ymax=None, latency=False):
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    ax.set_xlabel("Array Size (bytes)")
    if latency:
        ax.set_ylabel("Latency (ns)")
    else:
        ax.set_ylabel("Inverse throughput (ns)")

    global dashes
    # for s in data.Style.unique():
    # assert s in dashes

    data = data[data["name"].isin(names)]
    sns.lineplot(
        x="sz",
        y="latency",
        hue=data[
            "name"
            # "Color" if len(data.batchsize.unique()) == 1 else "display_name"
        ].tolist(),
        # style="Style",  # if data.Style.unique().tolist() != [""] else None,
        # dashes=dashes,
        data=data,
        legend="auto",
        sizes=[2, 3, 4, 5, 6, 7, 8, 9],
        palette=palette,
        errorbar=("pi", 100),
        estimator="median",
    )

    ax.set_xscale("log", base=2)
    # Add more xticks locator
    # ax.set_xticks()
    ax.xaxis.set_major_locator(LogLocator(base=4, numticks=20))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left")

    # Add vertical lines for cache sizes
    for size, name in [(l1_size, "L1"), (l2_size, "L2"), (l3_size, "L3")]:
        ax.axvline(x=size, color="red", linestyle="--", zorder=0)
        ax.text(size, 0, f"{name} ", color="red", va="bottom", ha="right")

    #
    if l3_size < data.sz.max():
        ax.text(data.sz.max(), 0, "RAM", color="red", va="bottom", ha="right")

    # Save
    fig.savefig(f"plots/{experiment_name}.png", bbox_inches="tight", dpi=600)
    print(f"Saved {experiment_name}.png")
    fig.savefig(f"plots/{experiment_name}.svg", bbox_inches="tight")
    print(f"Saved {experiment_name}.svg")
    return fig


def summary_table(data):
    # Table of statistics at a few key sizes.
    szs = [l1_size / 2, l2_size / 2, l3_size / 3, data.sz.max()]
    names = ["L1", "L2", "L3", "RAM"]
    data = data[data.sz.isin(szs)]
    table = pd.pivot_table(
        data,
        index="name",
        columns="display_size",
        values=["latency", "cycles"],
        sort=False,
    )
    print(
        tabulate.tabulate(
            table, headers=table.columns, tablefmt="orgtbl", floatfmt=".1f"
        )
    )


def display_size(size):
    if size <= l1_size:
        return "L1"
    if size <= l2_size:
        return "L2"
    if size <= l3_size:
        return "L3"
    return "RAM"


def read_file(filename):
    data = pd.read_json(filename)

    data["sz"] = data["size"]
    data["name"] = (
        data.scheme.str.replace("\\b[a-z_]+::", "", n=-1, regex=True)
        .str.replace("::{{closure}}", "")
        .str.replace("BTreeNode<16>::", "")
    )
    data["display_size"] = data["sz"].apply(display_size)

    # def style(name):
    #     if "prefetch" in name:
    #         return "Prefetch"
    #     if "latency" in name:
    #         return "Latency"
    #     else:
    #         return ""

    # def batchsize(name):
    #     if "batch" in name:
    #         return int(name.split("<")[1].split(">")[0])
    #     return 0

    # def color(name):
    #     name = name.split("<")[0]
    #     return name

    # data["Style"] = [style(name) for name in data.name]
    # data["batchsize"] = [batchsize(name) for name in data.name]
    # data["Color"] = [color(name) for name in data.display_name]

    global palette

    # names = sorted(data.display_name.unique())
    names = sorted(data.name.unique())
    colors = sns.color_palette(n_colors=10)
    colors = colors + colors + colors + colors
    palette = dict(zip(names, colors))
    # palette["Latency"] = "black"

    return data


# Read all files in the 'results' directory and iterate over them.
def plot_binary_search():
    data = read_file(f"results/results.json")
    all_names = data.name.unique().tolist()
    names = ["SortedVec::binary_search_std", "Eytzinger::search_prefetch<4>"]
    # plot(
    #     "1-binary-search",
    #     "Binary search and Eytzinger layout",
    #     data,
    #     names,
    #     latency=True,
    # ).show()

    names.remove("SortedVec::binary_search_std")
    names += [name for name in all_names if "search_with_find" in name]
    plot("2-find", "Optimizing the BTreeNode find function", data, names).show()

    # summary_table(data)


plt.style.use("dark_background")
plt.close("all")
plot_binary_search()
