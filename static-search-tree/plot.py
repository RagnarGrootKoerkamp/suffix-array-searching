#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tabulate
from matplotlib.ticker import LogLocator
import re

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
        style="Style",  # if data.Style.unique().tolist() != [""] else None,
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
    # fig.savefig(f"plots/{experiment_name}.png", bbox_inches="tight", dpi=600)
    # print(f"Saved {experiment_name}.png")
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
        # No type info here
        .str.replace("::{{closure}}", "")
        # Duplicated 16 from the STree
        .str.replace("BTreeNode<16>::", "")
        # The batched and full wrappers are not so interesting.
        .str.replace("(Batched|Full)<(\d+, )?STree<\d+, \d+>, (.*)>", "\\3", regex=True)
        # N is always 16; lets skip it.
        .str.replace("16, 16", "16")
    )
    # Cancat 'name' and 'index' columns.
    print(data.name.unique())
    print(data.params.unique())
    data.name = data.name + " " + data.params.astype(str)
    print(data.name.unique())
    data["display_size"] = data["sz"].apply(display_size)

    r = re.compile("<(\d+)>+$")

    def style(scheme):
        m = None
        m = re.search(r, scheme)
        if m is None:
            return ""
        var = m.group(1)
        return var

    # def batchsize(name):
    #     if "batch" in name:
    #         return int(name.split("<")[1].split(">")[0])
    #     return 0

    # def color(name):
    #     name = name.split("<")[0]
    #     return name

    data["Style"] = [style(scheme) for scheme in data.scheme]
    # data["batchsize"] = [batchsize(name) for name in data.name]
    # data["Color"] = [color(name) for name in data.display_name]

    global palette

    # names = sorted(data.display_name.unique())
    names = sorted(data.name.unique())
    colors = sns.color_palette(n_colors=10)
    colors = colors + colors + colors + colors + colors + colors
    palette = dict(zip(names, colors))
    # palette["Latency"] = "black"

    return data


# Read all files in the 'results' directory and iterate over them.
def plot_binary_search():
    data = read_file(f"results/results.json")
    all_names = data.name.unique().tolist()
    keep = []

    def prune(names):
        nonlocal all_names, keep
        for n in names:
            if n in all_names and n not in keep:
                all_names.remove(n)

    def select(select):
        nonlocal all_names
        if not isinstance(select, list):
            select = [select]
        selected = []
        for n in all_names:
            for s in select:
                if s in n:
                    selected.append(n)
                    break
        print(f"select {select} from {all_names} -> {selected}")
        return selected

    spare_names = select(["Rev", "Fwd"])
    prune(spare_names)

    names = ["SortedVec::binary_search_std", "Eytzinger::search_prefetch<4>"]
    # plot(
    #     "1-binary-search",
    #     "Binary search and Eytzinger layout",
    #     data,
    #     names,
    #     latency=True,
    # ).show()
    keep.append(names[-1])
    prune(names)

    names = keep + select("search_with_find<find_linear>")
    plot("2-hugepages", "Hugepages", data, names).show()
    prune(select("NoHuge"))

    names = keep + select("search_with_find")
    plot("3-find", "Optimizing the BTreeNode find function", data, names).show()
    prune(names)

    names = keep + select("batch<")
    plot("4-batching", "Batch size", data, names, ymax=120).show()
    keep += select("batch<128>")
    prune(names)

    names = keep + select(["batch_prefetch"])
    keep.append(names[-1])
    plot("5-prefetch", "Prefetching", data, names, ymax=60).show()
    prune(names)

    names = keep + select(
        ["batch_prefetch<128", "splat<128", "ptr<128", "ptr2<128", "ptr3<128"]
    )
    keep.append(names[-1])
    plot("6-improvements", "Improvements", data, names, ymax=30).show()
    prune(names)

    names = keep + select("skip_prefetch")
    plot("7-skip-prefetch", "Skip prefetch", data, names, ymax=30).show()
    prune(names)

    names = keep + select("interleave")
    plot("8-interleave", "Interleave", data, names, ymax=30).show()
    prune(names)

    # Add construction parameter variants
    all_names += spare_names

    names = keep + select("Rev")
    names = [n for n in names if "<15" not in n]
    plot("9-params", "Memory layout", data, names, ymax=30).show()
    keep.append(select("Rev")[0])
    prune(names)

    names = keep + select("<15")
    plot("10-base15", "Base 15", data, names, ymax=30).show()
    prune(names)

    plot("99-base15", "Remainder", data, all_names, ymax=30).show()
    # summary_table(data)


plt.style.use("dark_background")
plt.close("all")
plot_binary_search()
