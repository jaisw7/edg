from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from cycler import cycler
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


class ContourPlot:

    # fonts
    large = 12
    medium = 10
    small = 8

    # resolution
    dpi = 300

    # figure size
    figsize = (5, 5)

    # Define Tecplot-like RGB colors
    colors = [
        (0, 0, 1),  # Blue
        (0, 1, 1),  # Cyan
        (0, 1, 0),  # Green
        (1, 1, 0),  # Yellow
        (1, 0, 0),  # Red
    ]

    # major ticks
    major_ticks_width = 1
    major_ticks_length = 8

    # minor ticks
    minor_ticks_width = major_ticks_width / 2
    minor_ticks_length = major_ticks_length / 2

    def __init__(self, *args, **kwargs):
        rcParams["font.family"] = "Helvetica"
        rcParams["axes.labelweight"] = "bold"
        rcParams["axes.labelsize"] = self.medium
        rcParams["axes.labelpad"] = -0.5
        rcParams["lines.linewidth"] = 0.75
        rcParams["legend.fontsize"] = self.small
        rcParams["axes.prop_cycle"] = self.cycler

    @property
    def cmap(self):
        return LinearSegmentedColormap.from_list("tecplot", self.colors, N=128)

    @property
    def cycler(self):
        # fmt: off
        return cycler(
            "color",
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        )
        # fmt: on

    @property
    def formatter(self):
        return FuncFormatter(lambda v, t: f"{v:.6g}")

    def subplots(self, *args, **kwargs):
        kwargs.setdefault("figsize", self.figsize)
        kwargs.setdefault("dpi", self.dpi)
        fig, ax = plt.subplots(*args, **kwargs)
        self.ax, self.fig = ax, fig

        # set minor ticks
        ax.minorticks_on()
        ax.tick_params(
            which="both",
            length=self.minor_ticks_length,
            width=self.minor_ticks_width,
            color="black",
            direction="in",
        )

        # remove top and right border
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # set width of left and bottom border
        ax.spines["left"].set_linewidth(self.major_ticks_width)
        ax.spines["bottom"].set_linewidth(self.major_ticks_width)

        # Apply the formatter to both axes
        ax.xaxis.set_major_formatter(self.formatter)
        ax.yaxis.set_major_formatter(self.formatter)

        # set major ticks
        ax.tick_params(
            axis="both",
            direction="in",
            length=self.major_ticks_length,
            width=self.major_ticks_width,
            colors="black",
            labelsize=self.small,
        )

        return (fig, ax)

    def _ensure_plot(self):
        if not hasattr(self, "ax"):
            fig, ax = self.subplots()

    def contourf(self, *args, **kwargs):
        self._ensure_plot()
        kwargs.setdefault("cmap", self.cmap)
        return self.ax.contourf(*args, *kwargs)

    def savepdf(self, filename: str):
        plt.tight_layout()
        filename = Path(filename).with_suffix(".pdf")
        self.fig.savefig(
            filename,
            format="pdf",
            bbox_inches="tight",
            dpi=self.dpi,
            pad_inches=0,
        )

    def colorbar(
        self,
        contour,
        label,
        ticks,
        tick_width=0.2,
        format="%.8g",
        inset_params=[0.65, 1.05, 0.35, 0.05],
        label_pos=(-0.15, 0.0),
        label_pad=-8,
    ):
        cbar = self.fig.colorbar(
            contour,
            cax=self.ax.inset_axes(inset_params),
            orientation="horizontal",
            ticks=ticks,
            format=format,
        )
        cbar.ax.tick_params(
            axis="x",
            direction="in",
            width=tick_width,
            colors="black",
            labelsize=self.small,
        )
        cbar.ax.set_xlabel(
            label,
            fontsize=self.small,
            fontweight="bold",
            labelpad=label_pad,
            loc="left",
        )
        cbar.ax.get_xaxis().label.set_position(label_pos)
        cbar.outline.set_linewidth(tick_width)
        cbar.ax.spines["top"].set_visible(False)
        cbar.ax.spines["right"].set_visible(False)
        cbar.ax.spines["bottom"].set_visible(False)
        cbar.ax.spines["left"].set_visible(False)
        return cbar

    def tricontour(self, *args, **kwargs):
        self._ensure_plot()
        kwargs.setdefault("colors", "k")
        return self.ax.tricontour(*args, **kwargs)

    def tricontourf(self, *args, **kwargs):
        self._ensure_plot()
        kwargs.setdefault("cmap", self.cmap)
        return self.ax.tricontourf(*args, **kwargs)

    def get_triangulation_delaunay(self, mesh):
        x, y = mesh.points[:, 0], mesh.points[:, 1]
        triangles = mesh.delaunay_2d().faces.reshape(-1, 4)[:, 1:4]
        return (x, y, tri.Triangulation(x, y, triangles))

    def get_triangulation_vtk(self, mesh):
        x, y = mesh.points[:, 0], mesh.points[:, 1]
        triangles = mesh.cell_connectivity.reshape(-1, 3)
        return (x, y, tri.Triangulation(x, y, triangles))

    def add_rect(self, x, y, w, h, lc="k", fc="silver", lw=1, ls="solid", zo=2):
        self._ensure_plot()
        self.ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                edgecolor=lc,
                facecolor=fc,
                linewidth=lw,
                linestyle=ls,
                zorder=zo,
            )
        )

    def add_text(self, x, y, text, fw="bold", zo=2):
        self._ensure_plot()
        self.ax.text(
            x,
            y,
            text,
            fontsize=self.medium,
            ha="center",
            va="center",
            fontweight=fw,
            zorder=zo,
        )

    def plot(self, *args, **kwargs):
        self._ensure_plot()
        return self.ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self._ensure_plot()
        return self.ax.scatter(*args, **kwargs)

    def clear_lines(self):
        self._ensure_plot()
        self.ax.legend_.remove()
        for line in self.ax.lines:
            line.remove()
        self.ax.set_prop_cycle(self.cycler)

    def show(self):
        plt.show()
