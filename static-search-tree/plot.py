#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tabulate
from matplotlib.ticker import LogLocator
from matplotlib.colors import to_rgba
import re

import mpld3

# import plotly.io as pio
# import plotly
# import plotly.tools as tls

palette = None
dashes = {"": ""}


def caches():
    sizes = []
    # Note: index1 for me is the L1 instruction cache.
    # Note: All read strings are eg 32K.
    for i, name in [(0, "L1"), (2, "L2"), (3, "L3")]:
        t = Path(f"/sys/devices/system/cpu/cpu0/cache/index{i}/size").read_text()
        sizes.append((int(t[:-2]) * 1024, name))
    return sizes


(l1_size, _), (l2_size, _), (l3_size, _) = caches()


def plot(
    experiment_name,
    title,
    data,
    names,
    # Previous results are dimmed,
    # but the most resent one is bolded.
    keep,
    new_best=None,
    skip=0,
    ymax=None,
    latency=False,
    style=None,
    size=False,
):
    print("plotting", experiment_name, "\n", names)
    # Create a figure
    fig, ax = plt.subplots(figsize=(11, 6))
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

    new_best = None if new_best is False else new_best or names[-1]

    def type(name):
        if keep and name == keep[-1]:
            return "best"
        if name == new_best:
            return "new_best"
        if name in keep:
            return "old"
        return "new"

    data["type"] = data.name.apply(type)

    alpha_map = {"old": 0.5, "new": 1, "best": 1, "new_best": 1}
    my_palette = {name: to_rgba(palette[name], alpha_map[type(name)]) for name in names}

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
        size="type",
        sizes={"old": 0.5, "new": 1, "best": 2, "new_best": 1.5},
        palette=my_palette,
        errorbar=("pi", 50),
        estimator="median",
    )

    # Plot index size with a separate y-scale.
    if size:
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
        size_ax.grid(True, alpha=0.4, ls="dotted", lw=1)
        size_ax.set_ylim(2**0, 2**30)
        size_ax.set_ylabel("Datastructure size")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=4, numticks=20))
    ax.set_ylim(0, ymax)
    ax.grid(True, alpha=0.4)
    ax.grid(which="minor", color="gray", alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(lambda x: x / 4, lambda x: x * 4))
    secax.set_xscale("log", base=2)
    secax.set_xlabel("Array length (u32)")
    secax.set_xticks([4**i for i in range(20)])

    # Drop new/old/best from the legend.
    ax.legend(loc="upper left")
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
        if l not in ["new", "old", "best", "new_best", "type", "scheme"]:
            new_handles.append(h)
            new_labels.append(l)
    ax.legend(new_handles, new_labels, loc="upper left")

    # Add vertical lines for cache sizes
    for size, name in caches():
        ax.axvline(x=size, color="red", linestyle="--", zorder=0)
        ax.text(size, 0, f"{name} ", color="red", va="bottom", ha="right")

    if l3_size < data.sz.max():
        ax.text(data.sz.max(), 0, "RAM", color="red", va="bottom", ha="right")

    # Save
    # fig.savefig(f"plots/{experiment_name}.png", bbox_inches="tight", dpi=600)
    # print(f"Saved {experiment_name}.png")
    fig.savefig(f"plots/{experiment_name}.svg", bbox_inches="tight")
    print(f"Saved {experiment_name}.svg")

    # pio.write_html(fig, f"plots/{experiment_name}.html", auto_open=True)
    # Path(f"plots/{experiment_name}.html").write_text(mpld3.fig_to_html(fig))
    # plotly_fig = tls.mpl_to_plotly(fig)
    # plotly.offline.plot(plotly_fig, filename="plotly version of an mpl figure")

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


r1 = re.compile("\\b[a-z_]+::")
r2 = re.compile("(Batched|Full)<(\\d+, )?(Partitioned)?STree<[^<>]*>, (.*)>")


def clean_name(name):
    # strip module names
    name = re.sub(r1, "", name)
    # No type info here
    name = name.replace("::{{closure}}", "")
    # Duplicated 16 from the STree
    name = name.replace("BTreeNode<16>::", "")
    # Drop batch wrapper
    # The batched and full wrappers are not so interesting.
    name = re.sub(r1, "", name)
    name = re.sub(r2, "\\4", name)
    name = name.replace("STree<16, 16>", "STree<>")
    name = name.replace("STree<15, 16>", "STree<B=15>")
    name = name.replace("PartitionedSTree<16, 16, false, false>", "PartitionedSTree<>")
    name = name.replace(
        "PartitionedSTree<16, 16, true, false>", "PartitionedSTree<Compact>"
    )
    name = name.replace("PartitionedSTree<16, 16, false, true>", "PartitionedSTree<L1>")
    name = name.replace("search<128, false>", "search<>")
    name = name.replace("search<128, true>", "search<Prefetch>")
    name = name.replace("search_with_find<find_", "search<find_")
    name = name.replace("splat<128>", "splat")
    name = name.replace("ptr<128>", "ptr")
    name = name.replace("ptr2<128>", "ptr2")
    name = name.replace("ptr3<128>", "ptr3")
    name = name.replace("ptr3_full<128>", "ptr3_full")
    name = name.replace("skip_prefetch<128, ", "skip_prefetch<SKIP=")
    return name


def read_file(filename):
    data = pd.read_json(filename)

    data["sz"] = data["size"]
    data["scheme"] = data["scheme"].apply(clean_name)
    # Cancat 'name' and 'index' columns.
    data["name"] = (data.scheme + " " + data.params.astype(str)).str.strip()
    data["display_size"] = data["sz"].apply(display_size)

    data["overhead"] = data.index_size / data.sz

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
        keep,
        latency=True,
    ).show()
    keep += names

    names = keep + sel("search<find_linear>")
    plot("2-hugepages", "Stree and hugepages", data, names, keep).show()
    prune(sel("NoHuge"))

    names = keep + sel("search<find")
    plot(
        "3-find", "Optimizing the BTreeNode find function", data, names, keep, ymax=240
    ).show()
    keep.append(names[-1])
    prune(names)

    names = keep + sel("batch<")
    plot("4-batching", "Batch size", data, names, keep, ymax=120).show()
    keep += sel("batch<128>")
    prune(names)

    names = keep + sel(["batch_prefetch"])
    plot("5-prefetch", "Prefetching", data, names, keep, ymax=60).show()
    keep.append(names[-1])
    prune(names)

    names = keep + sel(["batch_prefetch<128", "splat", "ptr"])
    plot("6-improvements", "Improvements", data, names, keep, ymax=30).show()
    keep.append(names[-1])
    prune(names)

    names = keep + sel("skip_prefetch")
    plot("7-skip-prefetch", "Skip prefetch", data, names, keep, ymax=30).show()
    prune(names)

    names = keep + sel("interleave")
    plot("8-interleave", "Interleave", data, names, keep, ymax=30).show()
    prune(names)

    # Add construction parameter variants
    all_names += spare_names

    names = keep + select("STree<>", sel("Rev"))
    new_best = sel("Rev+Fwd")[0]
    plot(
        "9-params",
        "Memory layout",
        data,
        names,
        keep,
        new_best=new_best,
        ymax=30,
        size=True,
    ).show()
    keep.append(new_best)
    prune(names)

    names = keep + sel("<B=15")
    plot("10-base15", "Base 15", data, names, keep, new_best=False, ymax=30).show()
    prune(names)

    plot("11-summary", "Summary", data, keep, [], ymax=30).show()

    names = keep + sel("PartitionedSTree<>")
    new_best = names[-2]
    plot(
        "20-prefix",
        "Split by prefix",
        data,
        names,
        keep,
        new_best=new_best,
        ymax=30,
        size=True,
    ).show()
    prune(names)
    keep.append(new_best)

    names = keep + sel("PartitionedSTree<Compact>")
    new_best = False  # names[-2]
    plot(
        "21-compact",
        "Per-tree compact layout",
        data,
        names,
        keep,
        new_best=new_best,
        ymax=30,
        size=True,
    ).show()
    prune(names)
    # keep.append(new_best)

    names = keep + select("search<>", sel("PartitionedSTree<L1>"))
    new_best = names[-2]
    plot(
        "22-l1",
        "L1-compression",
        data,
        names,
        keep,
        new_best=new_best,
        ymax=30,
        size=True,
    ).show()
    prune(names)
    keep.pop()
    keep.pop()
    keep.append(new_best)

    # Drop the pruning variants; they're not interesting.
    prune(sel("<Prefetch>"))

    plot("23-summary", "Summary", data, keep, [], ymax=30).show()

    # plot("99", "Remainder", data, all_names, [], ymax=30).show()
    # summary_table(data)


def filter_large(data, x=2.5):
    data["latency"] = data.apply(
        lambda row: (row.latency if row.overhead < x else 0), axis=1
    )
    data = data[data.sz > 2**12]
    return data


def plot_all():
    data = read_file(f"results/results.json")
    data = filter_large(data, 1.3)
    names = data.name.unique().tolist()
    print("Plotting...")
    keep = select("ptr3", names)
    names2 = keep + select("PartitionedSTree<>", names)
    # plot("parts", "Parts", data, names2, ymax=30).show()
    names2 = keep + select("PartitionedSTree<Compact>", names)
    # plot("compact", "Compact parts", data, names2, ymax=30).show()
    names2 = keep + select("PartitionedSTree<L1>", names)
    plot("parts-l1", "Parts L1", data, names2, keep, ymax=30).show()


# plt.style.use("dark_background")
# plt.close("all")
plot_binary_search()
# plot_all()
