import inspect
import json
import re
import sys
import tomllib
from collections.abc import Mapping
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable, Literal, Optional

import emoji
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from matplotlib.lines import Line2D
from PIL import Image
from pydantic import BaseModel
from requests import Response, get
from requests.exceptions import (ConnectionError, HTTPError, RequestException,
                                 Timeout)
from scipy.fft import fft, fftfreq

from marn_x.utils.plot_styling import continuous_colored_title

logger.level("INFO")


class AllVars(Mapping):
    def __init__(self, **kwargs):
        if "file_stem" in kwargs:
            self.file_stem = kwargs["file_stem"]
        else:
            self.file_stem = Path(inspect.stack()[1][1]).stem

        # Global toml settings
        for key in config["settings"]:
            setattr(self, key, config["settings"][key])

        # Overwrite with file specific settings
        if self.file_stem in config:
            for key in config[self.file_stem]:
                setattr(self, key, config[self.file_stem][key])
        else:
            logger.warning(
                f"No config.toml entry for file {self.file_stem}, continuing..."
            )

        # Overwrite with any keyword arguments passed to init
        for key in kwargs:
            setattr(self, key, kwargs[key])

        logger.remove()
        logger.add(sys.stderr, level=self.logging_level)

        logger.info(f"Retreived {len(self)} variables for {self.file_stem}")

    def __str__(self) -> str:
        return str(vars(self))

    # Implement some standard methods needed before this class can be a mapping *sigh*
    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, key):
        return vars(self)[key]

    def __len__(self) -> int:
        return len(vars(self))


class GeneralSettings(BaseModel):
    logging_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ]
    raw: str
    processed: str
    models: str
    img: str
    message_col: str
    author_col: str
    timestamp_col: str

    file_stem: str
    input: dict[str, str]
    request_headers: Optional[dict] = None
    json_nests: Optional[list[str]] = None


class PlotSettings(GeneralSettings):
    """Base settings for plotting."""

    title: str
    xlabel: str
    ylabel: str
    figsize: tuple = (12, 6)
    title_fontsize: int = 8
    suptitle_fontweight: str = "bold"
    hide_spines: bool = True
    hide_axis_label: bool | str = False
    hide_axis: bool | str = False
    date_format_xaxis: Optional[str] = None
    date_format_yaxis: Optional[str] = None
    percentage: Optional[str] = None
    hide_legend: bool = True
    suptitle: Optional[str] = None
    suptitle_x: float = 0.5
    suptitle_y: float = 1.07
    suptitle_ha: str = "center"
    title_x: int | float = 0.5
    title_y: int | float = 1
    linewidth: int = 3
    annotation_fontsize: int = 8
    annotation_verticalalignment: str = "top"
    annotation_fontweight: str = "bold"
    annotations: Optional[list[dict[str, int | float | str | datetime]]] = None
    annotation_x: float = 1.0
    annotation_y: float = 1.0
    annotation_x_offset: float = 0.0
    annotation_y_offset: float = 0.0
    uber_suptitle: Optional[str] = None
    suptitle_parts: Optional[list[str]] = None
    suptitle_parts_x: float = 0.0
    suptitle_colors: list[str] = [""]
    suptitle_fontsize: int = 11
    legend_title: Optional[str] = None

    def __str__(self) -> str:
        return str(vars(self))


class BasePlot:
    settings: PlotSettings
    fig: plt.Figure | go.Figure
    ax: plt.Axes
    """Base class for creating plots."""

    def __init__(self, settings: PlotSettings):
        self.settings = settings
        self.fig, self.ax = plt.subplots(figsize=self.settings.figsize)

    def create_figure(self, **kwargs) -> tuple:
        """Create a figure and configure it based on settings."""

        for key in kwargs:
            setattr(self.settings, key, kwargs[key])

        # Load all variables from config.toml, for use in format strings in text (global first, then overwritten by variables per file).
        # If variables passed to function, they are used over config.toml.
        vars_settings = vars(self.settings)

        try:
            self.ax.set_xlabel(self.settings.xlabel.format(**vars_settings))

            self.ax.set_ylabel(self.settings.ylabel.format(**vars_settings))

            self.ax.set_title(
                self.settings.title.format(**vars_settings),
                fontsize=self.settings.title_fontsize,
                y=self.settings.title_y,
                x=self.settings.title_x,
            )
        except ValueError:
            logger.error(
                "Settings class or config.toml have to be set to plot xlabel, ylabel and title."
            )

        if self.settings.date_format_xaxis:
            self.ax.xaxis.set_major_formatter(
                mdates.DateFormatter(self.settings.date_format_xaxis)
            )
        if self.settings.date_format_yaxis:
            self.ax.yaxis.set_major_formatter(
                mdates.DateFormatter(self.settings.date_format_yaxis)
            )

        if self.settings.percentage:
            if self.settings.percentage in ("x", "xaxis", "both"):
                self.ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            if self.settings.percentage in ("y", "yaxis", "both"):
                self.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            if self.settings.percentage not in ("xaxis", "yaxis", "x", "y", "both"):
                raise ValueError("percentage has to be xaxis or yaxis...")

        if self.settings.uber_suptitle:

            self.fig.suptitle(
                self.settings.uber_suptitle.format(**vars_settings)
            ).set_fontweight(self.settings.suptitle_fontweight)

        if self.settings.suptitle_parts:

            continuous_colored_title(
                self.ax,
                [
                    part.format(**vars_settings) if isinstance(part, str) else part
                    for part in self.settings.suptitle_parts
                ],
                [
                    part.format(**vars_settings) if isinstance(part, str) else part
                    for part in self.settings.suptitle_colors
                ],
                y_position=self.settings.suptitle_y,
                fontsize=self.settings.suptitle_fontsize,
                start_x=self.settings.suptitle_parts_x,
            )

        if self.settings.suptitle:

            self.ax.text(
                self.settings.suptitle_x,
                self.settings.suptitle_y,
                self.settings.suptitle.format(**vars_settings),
                horizontalalignment=self.settings.suptitle_ha,
                transform=self.ax.transAxes,
                fontsize=self.settings.suptitle_fontsize,
            )

        if self.settings.annotations:
            for instance in self.settings.annotations:
                # Convert datetime inputs to float with built-in Matplotlib function
                instance["x"] = (
                    mdates.date2num(instance["x"])
                    if isinstance(instance["x"], datetime)
                    else instance["x"]
                )
                instance["y"] = (
                    mdates.date2num(instance["y"])
                    if isinstance(instance["y"], datetime)
                    else instance["y"]
                )
                self.ax.annotate(
                    str(instance["text"]),
                    (
                        (
                            self.settings.annotation_x
                            + self.settings.annotation_x_offset
                            if instance["x"] == "ANNOT_X"
                            else (
                                instance["x"]
                                if isinstance(instance["x"], int)
                                or isinstance(instance["x"], float)
                                else 0.0
                            )
                        ),
                        (
                            self.settings.annotation_y
                            + self.settings.annotation_y_offset
                            if instance["y"] == "ANNOT_Y"
                            else (
                                instance["y"]
                                if isinstance(instance["y"], int)
                                or isinstance(instance["y"], float)
                                else 0.0
                            )
                        ),
                    ),
                    fontsize=self.settings.annotation_fontsize,
                    verticalalignment=self.settings.annotation_verticalalignment,
                )

        if self.settings.hide_spines:
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.spines["bottom"].set_visible(False)
            self.ax.spines["left"].set_visible(False)

        if self.settings.hide_axis in ("x", "xaxis", "both", True):
            self.ax.get_xaxis().set_visible(False)
        if self.settings.hide_axis in ("y", "yaxis", "both", True):
            self.ax.get_yaxis().set_visible(False)

        if self.settings.hide_axis_label in ("x", "xaxis", "both", True):
            self.ax.get_xaxis().get_label().set_visible(False)
        if self.settings.hide_axis_label in ("y", "yaxis", "both", True):
            self.ax.get_yaxis().get_label().set_visible(False)

        if self.settings.legend_title:
            self.ax.legend(title=self.settings.legend_title)

        if self.settings.hide_legend and self.ax.get_legend():
            self.ax.get_legend().set_visible(False)

        self.fig.tight_layout()
        return self.fig, self.ax

    def get_figure(self, **kwargs) -> tuple:
        """Return the figure, creating it if needed."""
        if (not (self.ax.lines or self.ax.collections)) or len(kwargs) > 0:
            self.create_figure(**kwargs)
        return self.fig, self.ax

    def to_png(
        self,
        fig: Optional[go.Figure | plt.Figure] = None,
        path: Optional[Path] = None,
    ) -> Path:
        fig = fig or self.fig

        if fig is None:
            raise ValueError(f"No fig to save for {self.settings.file_stem}")

        if not path:
            path = (
                Path(__file__).parent
                / "../../.."
                / self.settings.processed
                / f"{self.settings.file_stem}-{get_now_str()}.png"
            ).resolve()

        if isinstance(fig, go.Figure):
            fig.write_image(path)
        elif isinstance(fig, plt.Figure):
            fig.savefig(path)

        logger.success(f"Succesfully saved plot to {path}")

        return path

    def to_html(
        self,
        fig: Optional[go.Figure] = None,
        path: Optional[Path] = None,
    ) -> Path:
        fig = fig or self.fig

        if fig is None:
            raise ValueError(f"No fig to save for {self.settings.file_stem}")

        if not isinstance(fig, go.Figure):
            raise ValueError("Only Plotly objects can be saved to HTML.")

        if not path:
            path = (
                Path(__file__).parent
                / "../../.."
                / self.settings.processed
                / f"{self.settings.file_stem}-{get_now_str()}.html"
            ).resolve()

        fig.write_html(str(path))

        logger.success(f"Succesfully saved plot to {path}")

        return path


def is_hyperlink(path: str | Path) -> bool:
    if isinstance(path, Path):
        return False
    return bool(
        re.search(
            r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$",
            str(path),
        )
    )


with open((Path(__file__).parent / "../../../config.toml").resolve(), "rb") as f:
    config = tomllib.load(f)
