#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tabulate
from matplotlib.ticker import LogLocator
import re

palette = None
dashes = {"": ""}

(l1_size, _), (l2_size, _), (l3_size, _) = caches()


def caches():
    sizes = []
    # Note: index1 for me is the L1 instruction cache.
    # Note: All read strings are eg 32K.
    for i, name in [(0, "L1"), (2, "L2"), (3, "L3")]:
        t = Path(f"/sys/devices/system/cpu/cpu0/cache/index{i}/size").read_text()
        sizes.append((int(t[:-2]) * 1024, name))
    return sizes


def plot(
    experiment_name, title, data, names, skip=0, ymax=None, latency=False, style="Style"
):
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
        style=style,  # if data.Style.unique().tolist() != [""] else None,
        # dashes=dashes,
        data=data,
        legend="auto",
        sizes=[2, 3, 4, 5, 6, 7, 8, 9],
        palette=palette,
        errorbar=("pi", 50),
        estimator="median",
    )

    # Plot index size with a separate y-scale.
    size_ax = ax.twinx()
    sns.lineplot(
        x="sz",
        y="index_size",
        hue=data["name"].tolist(),
        # style=style,
        linestyle="dotted",
        data=data,
        legend=None,
        palette=palette,
        ax=size_ax,
    )

    size_ax.set_yscale("log", base=2)
    size_ax.grid(True, alpha=0.4, ls="dotted")
    size_ax.set_ylim(2**6, 2**30)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=4, numticks=20))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.4)
    ax.grid(which="minor", color="gray", alpha=0.2)
    ax.legend(loc="upper left")

    secax = ax.secondary_xaxis("top", functions=(lambda x: x / 4, lambda x: x * 4))
    secax.set_xscale("log", base=2)
    secax.set_xlabel("Array length (u32)")
    secax.set_xticks([4**i for i in range(20)])

    # Add vertical lines for cache sizes
    for size, name in caches():
        ax.axvline(x=size, color="red", linestyle="--", zorder=0)
        ax.text(size, 0, f"{name} ", color="red", va="bottom", ha="right")

    if l3_size < data.sz.max():
        ax.text(data.sz.max(), 0, "RAM", color="red", va="bottom", ha="right")

    # Save
    # fig.savefig(f"plots/{experiment_name}.png", bbox_inches="tight", dpi=600)
    # print(f"Saved {experiment_name}.png")
    # fig.savefig(f"plots/{experiment_name}.svg", bbox_inches="tight")
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
    data["scheme"] = (
        data.scheme.str.replace("\\b[a-z_]+::", "", n=-1, regex=True)
        # No type info here
        .str.replace("::{{closure}}", "")
        # Duplicated 16 from the STree
        .str.replace("BTreeNode<16>::", "")
        # The batched and full wrappers are not so interesting.
        .str.replace(
            "(Batched|Full)<(\\d+, )?(Partitioned)?STree<[^<>]*>, (.*)>",
            "\\4",
            regex=True,
        )
        # N is always 16; lets skip it.
        .str.replace("16, 16", "16")
    )
    # Cancat 'name' and 'index' columns.
    print(data.scheme.unique())
    print(data.params.unique())
    data["name"] = (data.scheme + " " + data.params.astype(str)).str.strip()
    print(data.name.unique())
    data["display_size"] = data["sz"].apply(display_size)

    r = re.compile("<(\\d+)>+$")

    def style(scheme):
        m = None
        m = re.search(r, scheme)
        if m is None:
            return ""
        var = m.group(1)
        return var

    data["Style"] = [style(scheme) for scheme in data.scheme]

    global palette

    names = sorted(data.name.unique())
    colors = sns.color_palette(n_colors=10)
    colors = colors + colors + colors + colors + colors + colors
    palette = dict(zip(names, colors))

    return data


def select(select, all_names):
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


# Read all files in the 'results' directory and iterate over them.
def plot_binary_search():
    data = read_file(f"results/results-release.json")
    all_names = data.name.unique().tolist()
    keep = []

    print(all_names)

    def prune(names):
        nonlocal all_names, keep
        for n in names:
            if n in all_names and n not in keep:
                all_names.remove(n)

    def sel(name):
        nonlocal all_names
        return select(name, all_names)

    spare_names = sel(["Rev", "Fwd"])
    prune(spare_names)

    names = sel(["binary_search", "Eytzinger"])
    plot(
        "1-binary-search",
        "Binary search and Eytzinger layout",
        data,
        names,
        latency=True,
    ).show()
    keep += names

    names = keep + sel("search_with_find<find_linear>")
    plot("2-hugepages", "Stree and hugepages", data, names).show()
    prune(sel("NoHuge"))

    names = keep + sel("search_with_find")
    plot(
        "3-find", "Optimizing the BTreeNode find function", data, names, ymax=240
    ).show()
    prune(names)

    names = keep + sel("batch<")
    plot("4-batching", "Batch size", data, names, ymax=120).show()
    keep += sel("batch<128>")
    prune(names)

    names = keep + sel(["batch_prefetch"])
    keep.append(names[-1])
    plot("5-prefetch", "Prefetching", data, names, ymax=60).show()
    prune(names)

    names = keep + sel(
        ["batch_prefetch<128", "splat<128", "ptr<128", "ptr2<128", "ptr3<128"]
    )
    keep.append(names[-1])
    plot("6-improvements", "Improvements", data, names, ymax=30).show()
    prune(names)

    names = keep + sel("skip_prefetch")
    plot("7-skip-prefetch", "Skip prefetch", data, names, ymax=30).show()
    prune(names)

    names = keep + sel("interleave")
    plot("8-interleave", "Interleave", data, names, ymax=30).show()
    prune(names)

    # Add construction parameter variants
    all_names += spare_names

    names = keep + sel("Rev")
    names = [n for n in names if "<15" not in n]
    plot("9-params", "Memory layout", data, names, ymax=30).show()
    keep.append(sel("Rev")[0])
    prune(names)

    names = keep + sel("<15")
    plot("10-base15", "Base 15", data, names, ymax=30).show()
    prune(names)

    plot("11-summary", "Summary", data, keep, ymax=30).show()

    plot("99", "Remainder", data, all_names, ymax=30).show()
    # summary_table(data)


def filter_large(data, x=2.5):
    data["latency"] = data.apply(
        lambda row: (row.latency if row["index_size"] < x * row.sz else 0), axis=1
    )
    data = data[data.sz > 2**12]
    return data


def plot_all():
    data = read_file(f"results/results.json")
    data = filter_large(data, 1.3)
    names = data.name.unique().tolist()
    keep = select("ptr3", names)
    names2 = keep + select("16, false, false", names)
    plot("parts", "Parts", data, names2, style="scheme", ymax=30).show()
    names2 = keep + select("16, true, false", names)
    plot("compact", "Compact parts", data, names2, style="scheme", ymax=30).show()
    names2 = keep + select("16, false, true", names)
    plot("parts-l1", "Parts L1", data, names2, style="scheme", ymax=30).show()


plt.style.use("dark_background")
plt.close("all")
# plot_binary_search()
plot_all()
