"""Rebuild the five-page report from daily returns and epoch histories."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np


REPORT_ORDER = ("cnn", "fourier")
ARCH_NAMES = {"cnn": "CNN + Transformer", "fourier": "Fourier + FFN"}
ARM_NAMES = ("Signal only", "Previous epoch", "Stopped gradients", "Full gradients")
POLICY_NAMES = ("Full data: epoch 100", "Full data: selected epoch",
                "Short window: epoch 100", "Rolling-selected epoch")
COLORS = {"fourier": ("#3333B2", "#117EAA", "#3D9C6A", "#CBB500"),
          "cnn": ("#C72535", "#FF9700", "#F5E100", "#39E600")}
SEED_COLORS = {"fourier": COLORS["fourier"],
               "cnn": ("#C72535", "#FF8C00", "#E6D600", "#39E600")}
INK = "#26313F"
NET_LIMITS = {"fourier": (.75, 2.), "cnn": (.875, 1.625)}
METRIC_NOTE = "Mean row: arithmetic average of seed SR, CAGR and Max DD. Curve: pointwise mean wealth."


class Page:
    """Place a seven-inch-wide figure at native size on a Letter page."""

    def __init__(self, height=10):
        self.fig = plt.figure(figsize=(8.5, 11))
        self.bottom = (11 - height) / 2

    def text(self, x, y, value, size=7.5, **kwargs):
        return self.fig.text((x + .75) / 8.5, (y + self.bottom) / 11, value,
                             fontsize=size, color=INK, **kwargs)

    def axes(self, x, y, width, height):
        return self.fig.add_axes([(x + .75) / 8.5, (y + self.bottom) / 11,
                                 width / 8.5, height / 11])

    def legend(self, handles, x, y, **kwargs):
        self.fig.legend(handles=handles, loc="center",
                        bbox_to_anchor=((x + .75) / 8.5, (y + self.bottom) / 11),
                        frameon=False, **kwargs)


def performance(data):
    """Average each seed metric separately from the pointwise mean wealth curve."""
    curves, metrics = {}, {}
    for architecture in REPORT_ORDER:
        a = data["architecture"].tolist().index(architecture)
        for arm in range(len(data["arm"])):
            for policy in range(len(data["policy"])):
                for basis in ("gross", "net"):
                    returns = data[basis][:, a, arm, policy]
                    paths = np.concatenate((np.ones((len(returns), 1)),
                                            np.cumprod(1 + returns, axis=1)), axis=1)
                    sharpe = np.sqrt(252) * returns.mean(axis=1) / returns.std(axis=1, ddof=1)
                    cagr = paths[:, -1] ** (252 / returns.shape[1]) - 1
                    drawdown = (1 - paths / np.maximum.accumulate(paths, axis=1)).max(axis=1)
                    seed_metrics = np.column_stack((sharpe, cagr, drawdown))
                    key = str(architecture), arm, policy, basis
                    # Mean statistics summarize seeds, not the averaged wealth trajectory.
                    curves[key] = np.vstack((paths, paths.mean(axis=0)))
                    metrics[key] = np.vstack((seed_metrics, seed_metrics.mean(axis=0)))
    dates = data["dates"].astype("datetime64[D]")
    dates = np.insert(dates, 0, np.busday_offset(dates[0], -1))
    return dates, curves, metrics


def seed_legend(page, y):
    page.legend([Line2D([], [], color=INK, lw=.7, alpha=.42, label="Seeds 0, 1, 2 (light)"),
                 Line2D([], [], color=INK, lw=1, label="Mean wealth curve")],
                3.5, y, ncol=2, fontsize=7.2, handlelength=2.5)


def table(page, x, y, width, key, label, color, metrics, cagr_label="CAGR"):
    page.text(x + .02, y + .595, label, size=7.4, weight="bold")
    ax = page.axes(x, y, width, .55)
    ax.set_axis_off()
    rows = [[seed, f"{sr:.2f}" if abs(sr) >= .005 else "0.00", f"{cagr:.2%}", f"{dd:.2%}"]
            for seed, (sr, cagr, dd) in zip(("0", "1", "2", "Mean"), metrics[key])]
    grid = ax.table(cellText=rows, colLabels=["Seed", "SR", cagr_label, "Max DD"],
                    colWidths=[.22, .21, .28, .29], cellLoc="right", bbox=(0, 0, 1, 1))
    grid.auto_set_font_size(False)
    grid.set_fontsize(7.1)
    for (row, column), cell in grid.get_celld().items():
        cell.set_linewidth(0)
        cell.set_text_props(color=INK)
        cell.PAD = .04
        if row == 0:
            cell.set_facecolor("#EFF1F4")
        if row == 4:
            cell.set_text_props(weight="bold", color=color)


def draw_paths(ax, dates, curves, keys, colors, seed_colors, limits, top=False):
    for key, color, seed_color in zip(keys, colors, seed_colors):
        for path in curves[key][:-1]:
            ax.plot(dates, path, color=seed_color, lw=.55, alpha=.42)
        ax.plot(dates, curves[key][-1], color=color, lw=1)
    ax.axhline(1, color="#A9B1BD", lw=.5)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(*limits)
    ax.set_xticks([np.datetime64(f"{year}-01-01") for year in (2011, 2015, 2020, 2025)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=7.4, length=2, width=.5, pad=2, labelbottom=not top)
    ax.grid(axis="y", color="#E1E5EC", lw=.45)
    ax.set_axisbelow(True)


def headline(dates, curves, metrics):
    page = Page(5.6)
    page.text(.42, 5.38, "CNN + Transformer: training and trading costs", size=11, weight="bold")
    seed_legend(page, 5.17)
    for panel, basis in enumerate(("gross", "net")):
        x = .42 + panel * 3.38
        policy = 1 if basis == "gross" else 3
        selection = "initial" if basis == "gross" else "rolling"
        keys = [("cnn", 0, 0, basis), ("cnn", 1, 0, basis), ("cnn", 3, policy, basis)]
        labels = ["Signal only, epoch 100", "Previous epoch, epoch 100",
                  f"Full gradients, {selection} validation"]
        colors = [COLORS["cnn"][i] for i in (0, 1, 3)]
        page.text(x, 4.96, "Costs off" if panel == 0 else "Costs on", size=9, weight="bold")
        page.text(x, 4.81, "Gross excess" if panel == 0 else "Net excess", size=7.1)
        for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
            table(page, x, 1.90 - i * .70, 3.08, key, label, color, metrics,
                  cagr_label="Excess CAGR")
        ax = page.axes(x, 2.76, 3.08, 1.92)
        draw_paths(ax, dates, curves, keys, colors, colors, (.875, 2.2))
        ax.set_yticks([1, 1.25, 1.5, 1.75, 2, 2.2])
        if panel:
            ax.tick_params(labelleft=False)
    page.text(.075, 3.72, "Excess return index (start = 1)", size=7.8,
              rotation=90, ha="center", va="center")
    page.text(.42, .36, "Red/orange: epoch 100. Green: full gradients, with initial validation on the left and rolling validation on the right.", size=6)
    page.text(.42, .23, "Green uses different selected models across panels. Costs on: 1 bp turnover + 0.1 bp/day short exposure.", size=6)
    page.text(.42, .10, METRIC_NOTE.replace("CAGR", "excess CAGR"), size=6)
    return page


def training_grid(architecture, dates, curves, metrics):
    page = Page()
    page.text(.42, 9.79, f"{ARCH_NAMES[architecture]}: training variants", size=11, weight="bold")
    seed_legend(page, 9.60)
    limits = NET_LIMITS[architecture]
    for policy, title in enumerate(POLICY_NAMES):
        row, column = divmod(policy, 2)
        x, top = (.42 if column == 0 else 3.80), (9.42 if row == 0 else 4.89)
        page.text(x, top, f"{chr(97 + policy)}  {title}", size=8.4, weight="bold", va="top")
        keys = [(architecture, arm, policy, "net") for arm in range(4)]
        for arm, key in enumerate(keys):
            table(page, x + (arm % 2) * 1.59, top - .83 - (arm // 2) * .68,
                  1.49, key, ARM_NAMES[arm], COLORS[architecture][arm], metrics)
        ax = page.axes(x + .10, top - 4.39, 2.98, 2.80)
        draw_paths(ax, dates, curves, keys, COLORS[architecture], SEED_COLORS[architecture], limits, row == 0)
        step = .25 if architecture == "fourier" else .125
        ax.set_yticks(np.arange(np.ceil(limits[0] / step), np.floor(limits[1] / step) + 1) * step)
        if column:
            ax.tick_params(labelleft=False)
    page.text(.075, 5, "Net excess return index (start = 1)", size=7.8,
              rotation=90, ha="center", va="center")
    page.text(.42, .20, METRIC_NOTE, size=6.2)
    page.text(.42, .075, "Net panels use shared y-axis limits; some seed tails extend beyond the axes. Statistics use every observation.", size=6.2)
    return page


def initial_validation(page, scores, origin):
    page.text(.42, origin + 4.58, "Initial validation: scores by epoch", size=10, weight="bold")
    page.text(.42, origin + 4.40, "720 training days, then 250 validation days (6 Jan-31 Dec 2010)", size=7.1)
    colors = ("#3333B2", "#D56A1D", "#14876D")
    for row, arm in enumerate((0, 3)):
        for column, architecture in enumerate(REPORT_ORDER):
            x, top = (.42 if column == 0 else 3.80), origin + (4.18 if row == 0 else 2.08)
            page.text(x, top, f"{ARCH_NAMES[architecture]}: {ARM_NAMES[arm].lower()}", size=8.2, weight="bold")
            ax = page.axes(x, top - 1.83, 3.08, 1.49)
            handles = []
            for seed, color in enumerate(colors):
                values = scores[architecture][seed, arm, 0]
                chosen = int(values.argmax())
                ax.plot(np.arange(1, 101), values, color=color, lw=.7, alpha=.5)
                ax.plot(chosen + 1, values[chosen], "o", color=color, ms=3, mec="white", mew=.4)
                handles.append(Line2D([], [], color=color, lw=.7, label=f"{seed}: epoch {chosen + 1}"))
            page.legend(handles, x + 1.54, top - .18, ncol=3, fontsize=6.8, handlelength=1.1, columnspacing=.8)
            ax.plot(np.arange(1, 101), scores[architecture][:, arm, 0].mean(axis=0), color=INK, lw=.85)
            ax.set(xlim=(1, 100), ylim=(-1.5, 5 if architecture == "cnn" else 2.25), xticks=[1, 20, 40, 60, 80, 100])
            ax.tick_params(labelsize=7, length=2, pad=2, labelbottom=row == 1)
            ax.grid(axis="y", color="#E1E5EC", lw=.45)
            ax.axhline(0, color="#A9B1BD", lw=.5)
    page.text(.075, origin + 2.4, "Annualized validation net Sharpe", size=7.5, rotation=90, ha="center", va="center")
    page.text(3.65, origin, "Training epoch", size=7, ha="center")


def rolling_validation(page, scores, seed, origin):
    page.text(.42, origin + 4.58, f"Rolling validation: seed {seed}", size=10, weight="bold")
    page.legend([Line2D([], [], marker="o", ms=3, mfc="white", mec=INK, color="none", label="Rolling selected epoch"),
                 Line2D([], [], color="#FF9157", lw=1.2, label="Initial constant epoch")],
                3.36, origin + 4.40, ncol=2, fontsize=7)
    for row, arm in enumerate((0, 3)):
        for column, architecture in enumerate(REPORT_ORDER):
            x, top = (.42 if column == 0 else 3.52), origin + (4.18 if row == 0 else 2.08)
            values = scores[architecture][seed, arm].T  # Epochs vertically, refits horizontally.
            chosen = values.argmax(axis=0) + 1
            page.text(x, top, f"{ARCH_NAMES[architecture]}: {ARM_NAMES[arm].lower()}", size=8.1, weight="bold")
            page.text(x, top - .15, f"Initial epoch {chosen[0]}; 31 refits, 100 epochs", size=6.8)
            ax = page.axes(x, top - 1.80, 2.72, 1.51)
            heat = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                             extent=(.5, 31.5, .5, 100.5), cmap="viridis", vmin=-3, vmax=5)
            ax.axhline(chosen[0], color="#FF9157", lw=1)
            ax.plot(np.arange(1, 32), chosen, linestyle="none", marker="o", ms=2.8,
                    mfc="white", mec=INK, mew=.35)
            ax.set(xticks=[1, 5, 10, 15, 20, 25, 31], yticks=[1, 25, 50, 75, 100])
            ax.tick_params(labelsize=7, length=2, pad=2, labelbottom=row == 1)
    bar = page.fig.colorbar(heat, cax=page.axes(6.42, origin + .28, .10, 3.61), ticks=[-3, -1, 1, 3, 5])
    bar.set_label("Validation net Sharpe", fontsize=7.8, labelpad=5)
    bar.ax.tick_params(labelsize=7.3, length=2)
    bar.outline.set_visible(False)
    page.text(.075, origin + 2.4, "Training epoch", size=7.5, rotation=90, ha="center", va="center")
    page.text(3.2, origin, "Refit", size=7, ha="center")


def make_report(results_dir, output_dir):
    """Write results.pdf, figures/headline.png and metrics.csv; no model fitting."""
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    with np.load(results_dir / "daily.npz") as data:
        dates, curves, metrics = performance(data)
        arms, policies = data["arm"].tolist(), data["policy"].tolist()
    with np.load(results_dir / "history.npz") as history:
        sharpe = history["metric"].tolist().index("sharpe")
        validation = history["values"][:, :, :, 1, :, :, 1, sharpe]
        # Display order is independent of the recorded architecture axis.
        scores = {str(architecture): validation[:, index]
                  for index, architecture in enumerate(history["architecture"])}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#7A8290", "axes.linewidth": .5,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    with PdfPages(output_dir / "results.pdf", metadata={"Title": "DLSA performance and training diagnostics"}) as pdf:
        page = headline(dates, curves, metrics)
        page.fig.savefig(output_dir / "figures/headline.png", dpi=400,
                         bbox_inches=Bbox.from_bounds(.75, page.bottom, 7, 5.6))
        pdf.savefig(page.fig)
        plt.close(page.fig)
        for architecture in REPORT_ORDER:
            page = training_grid(architecture, dates, curves, metrics)
            pdf.savefig(page.fig)
            plt.close(page.fig)
        for number in (1, 2):
            page = Page()
            page.text(.42, 9.83, f"Training diagnostics {number} of 2", size=7.5)
            if number == 1:
                initial_validation(page, scores, 5.05)
                rolling_validation(page, scores, 0, .25)
            else:
                rolling_validation(page, scores, 1, 5.05)
                rolling_validation(page, scores, 2, .25)
            page.text(.42, .075, "720 training + 250 validation days per refit. All epochs and refits retained. Heatmap scale: -3 to 5.", size=6.2)
            pdf.savefig(page.fig)
            plt.close(page.fig)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["architecture", "arm", "policy", "basis", "seed", "sharpe", "cagr", "max_drawdown"])
        for (architecture, arm, policy, basis), values in metrics.items():
            for seed, row in zip((0, 1, 2, "Mean"), values):
                writer.writerow([architecture, arms[arm], policies[policy], basis, seed, *row])
