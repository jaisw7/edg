from pathlib import Path

import matplotlib.pyplot as plt
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

    @property
    def cmap(self):
        return LinearSegmentedColormap.from_list("tecplot", self.colors, N=128)

    @property
    def formatter(self):
        return FuncFormatter(lambda v, t: f"{v:.2g}")

    def subplots(self, *args, **kwargs):
        kwargs.update({"figsize": self.figsize, "dpi": self.dpi})
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

    def contourf(self, *args, **kwargs):
        if not hasattr(self, "ax"):
            fig, ax = self.subplots()
        kwargs.update({"cmap": self.cmap})
        return self.ax.contourf(*args, *kwargs)

    def savepdf(self, filename: str):
        plt.tight_layout()
        filename = Path(filename).with_suffix(".pdf")
        self.fig.savefig(filename, format="pdf", bbox_inches="tight")

    def colorbar(
        self,
        contour,
        label,
        ticks,
        tick_width=0.2,
        format="%.2g",
        inset_params=[0.65, 1.05, 0.4, 0.05],
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
            length=14,
            width=tick_width,
            colors="black",
            labelsize=self.small,
        )
        cbar.ax.set_xlabel(
            label,
            fontsize=self.small,
            fontweight="bold",
            labelpad=-8,
            loc="left",
        )
        cbar.ax.get_xaxis().label.set_position((-0.15, 0.0))
        cbar.outline.set_linewidth(tick_width)
        cbar.ax.spines["top"].set_visible(False)
        cbar.ax.spines["right"].set_visible(False)
        cbar.ax.spines["bottom"].set_visible(False)
        cbar.ax.spines["left"].set_visible(False)
        return cbar
