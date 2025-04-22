"""
settings.py - Configuration and settings management for data visualization

This module provides classes and utilities for managing plot settings, configuration,
and creating standardized visualizations with matplotlib and plotly.

Classes:
    AllVars: Dictionary-like class for accessing configuration variables
    GeneralSettings: Base Pydantic model for general application settings
    PlotSettings: Extended settings model specifically for plot configuration
    BasePlot: Base class for creating and managing plots with consistent styling

Functions:
    is_hyperlink: Check if a string is a valid URL
    get_now_str: Get current datetime as a formatted string
"""

import inspect
import re
import sys
import tomllib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from loguru import logger
from pydantic import BaseModel

from marn_x.utils.plot_styling import continuous_colored_title

# Set default logging level
logger.level("INFO")


class AllVars(Mapping):
    """
    A dictionary-like class that aggregates configuration variables from multiple sources.

    This class loads variables from the global config.toml file, file-specific settings,
    and any provided keyword arguments. Priority order (highest to lowest):
    1. Keyword arguments passed to __init__
    2. File-specific settings from config[file_stem]
    3. Global settings from config["settings"]

    Implements the Mapping ABC to allow dictionary-like access to attributes.
    """

    def __init__(self, **kwargs):
        # Determine the source file name if not provided
        if "file_stem" in kwargs:
            self.file_stem = kwargs["file_stem"]
        else:
            # Use the name of the file that called this constructor
            self.file_stem = Path(inspect.stack()[1][1]).stem

        # Load global settings from config.toml
        for key in config["settings"]:
            setattr(self, key, config["settings"][key])

        # Override with file-specific settings if available
        if self.file_stem in config:
            for key in config[self.file_stem]:
                setattr(self, key, config[self.file_stem][key])
        else:
            logger.warning(
                f"No config.toml entry for file {self.file_stem}, continuing..."
            )

        # Override with any keyword arguments passed to init (highest priority)
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Configure logging based on settings
        logger.remove()
        logger.add(sys.stderr, level=self.logging_level)
        logger.add(
            (Path(__file__) / "../../../logs/logs.log"),
            rotation="weekly",
            level=self.logging_level,
        )

        logger.info(f"Retrieved {len(self)} variables for {self.file_stem}")

    def __str__(self) -> str:
        """String representation showing all variables as a dictionary."""
        return str(vars(self))

    # Implement required methods for Mapping ABC
    def __iter__(self):
        """Allow iteration over all attributes."""
        return iter(vars(self))

    def __getitem__(self, key):
        """Allow dictionary-style access to attributes."""
        return vars(self)[key]

    def __len__(self) -> int:
        """Return the number of attributes."""
        return len(vars(self))


class GeneralSettings(BaseModel):
    """
    Base model for application settings using Pydantic for validation.

    Contains core settings needed across the application, including
    file paths, logging configuration, and data column identifiers.
    """

    # Logging configuration
    logging_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ]

    # File paths for data and outputs
    raw: str  # Path to raw data
    processed: str  # Path to processed data
    models: str  # Path to models
    img: str  # Path to output images

    # Data column identifiers
    message_col: str  # Column name for message content
    author_col: str  # Column name for author/sender
    timestamp_col: str  # Column name for timestamp

    # File and input configuration
    file_stem: str  # Name of the current file without extension
    input: dict[str, str]  # Input data sources
    request_headers: Optional[dict] = None  # HTTP headers for API requests
    json_nests: Optional[list[str]] = None  # Nested JSON paths to extract


class PlotSettings(GeneralSettings):
    """
    Extended settings for plotting, inheriting from GeneralSettings.

    Includes comprehensive configuration options for plot appearance,
    titles, axes, annotations, and formatting.
    """

    # Basic plot properties
    title: str  # Main plot title
    xlabel: str  # X-axis label
    ylabel: str  # Y-axis label
    figsize: tuple = (12, 6)  # Figure size in inches (width, height)

    # Title and text styling
    title_fontsize: int = 8
    suptitle_fontweight: str = "bold"

    # Visibility toggles
    hide_spines: bool = True  # Whether to hide plot spines
    hide_axis_label: bool | str = False  # Hide specific axis labels ('x', 'y', 'both')
    hide_axis: bool | str = False  # Hide specific axes ('x', 'y', 'both')
    hide_legend: bool = True  # Whether to hide the legend

    # Date and number formatting
    date_format_xaxis: Optional[str] = None  # Format string for dates on x-axis
    date_format_yaxis: Optional[str] = None  # Format string for dates on y-axis
    percentage: Optional[str] = None  # Format axis as percentage ('x', 'y', 'both')

    # Super title (figure-level title) settings
    suptitle: Optional[str] = None  # Super title text
    suptitle_x: float = 0.5  # Horizontal position (0-1)
    suptitle_y: float = 1.07  # Vertical position
    suptitle_ha: str = "center"  # Horizontal alignment

    # Main title positioning
    title_x: int | float = 0.5  # Horizontal position (0-1)
    title_y: int | float = 1  # Vertical position

    # Line and annotation styling
    linewidth: int = 3  # Width of plot lines
    annotation_fontsize: int = 8  # Font size for annotations
    annotation_verticalalignment: str = "top"  # Vertical alignment of annotations
    annotation_fontweight: str = "bold"  # Font weight for annotations
    annotations: Optional[list[dict[str, int | float | str | datetime]]] = (
        None  # List of annotations
    )
    annotation_x: float = 1.0  # Default x position for annotations
    annotation_y: float = 1.0  # Default y position for annotations
    annotation_x_offset: float = 0.0  # Offset for x position
    annotation_y_offset: float = 0.0  # Offset for y position

    # Advanced title settings for multi-colored titles
    uber_suptitle: Optional[str] = None  # Alternative super title
    suptitle_parts: Optional[list[str]] = None  # Parts of segmented super title
    suptitle_parts_x: float = 0.0  # Starting x position for segmented title
    suptitle_colors: list[str] = [""]  # Background colors for title segments
    suptitle_fontsize: int = 11  # Font size for super title

    # Legend settings
    legend_title: Optional[str] = None  # Title for the legend

    def __str__(self) -> str:
        """String representation showing all settings as a dictionary."""
        return str(vars(self))


class BasePlot:
    """
    Base class for creating standardized plots with consistent styling.

    Provides methods for creating, configuring, and saving plots based on
    the provided settings. Supports both matplotlib and plotly figures.

    Attributes:
        settings: PlotSettings instance with configuration
        fig: matplotlib or plotly Figure object
        ax: matplotlib Axes object
    """

    settings: PlotSettings
    fig: plt.Figure | go.Figure
    ax: plt.Axes

    def __init__(self, settings: PlotSettings):
        """
        Initialize the plot with settings.

        Args:
            settings: PlotSettings object with plot configuration
        """
        self.settings = settings
        self.fig, self.ax = plt.subplots(figsize=self.settings.figsize)

    def create_figure(self, **kwargs) -> tuple:
        """
        Create a figure and configure it based on settings.

        Updates settings with any provided keyword arguments before creating the figure.
        Applies all configured styling, labels, titles, and annotations.

        Args:
            **kwargs: Additional or override settings

        Returns:
            tuple: (figure, axes) - The created figure and axes objects
        """
        # Update settings with any keyword arguments
        for key in kwargs:
            setattr(self.settings, key, kwargs[key])

        # Load all variables from settings, for use in format strings
        # (can be used like {variable_name} in text strings)
        vars_settings = vars(self.settings)

        # Set basic plot elements (labels and title)
        try:
            # Apply format strings to labels and title
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

        # Configure date formatting for axes if specified
        if self.settings.date_format_xaxis:
            self.ax.xaxis.set_major_formatter(
                mdates.DateFormatter(self.settings.date_format_xaxis)
            )
        if self.settings.date_format_yaxis:
            self.ax.yaxis.set_major_formatter(
                mdates.DateFormatter(self.settings.date_format_yaxis)
            )

        # Configure percentage formatting for axes if specified
        if self.settings.percentage:
            if self.settings.percentage in ("x", "xaxis", "both"):
                self.ax.xaxis.set_major_formatter(mtick.PercentFormatter())
            if self.settings.percentage in ("y", "yaxis", "both"):
                self.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            if self.settings.percentage not in ("xaxis", "yaxis", "x", "y", "both"):
                raise ValueError("percentage has to be xaxis or yaxis...")

        # Add super title (uber_suptitle) if specified
        if self.settings.uber_suptitle:
            self.fig.suptitle(
                self.settings.uber_suptitle.format(**vars_settings)
            ).set_fontweight(self.settings.suptitle_fontweight)

        # Add multi-colored title segments if specified
        if self.settings.suptitle_parts:
            continuous_colored_title(
                self.ax,
                # Format each part with variables if it's a string
                [
                    part.format(**vars_settings) if isinstance(part, str) else part
                    for part in self.settings.suptitle_parts
                ],
                # Format each color with variables if it's a string
                [
                    part.format(**vars_settings) if isinstance(part, str) else part
                    for part in self.settings.suptitle_colors
                ],
                y_position=self.settings.suptitle_y,
                fontsize=self.settings.suptitle_fontsize,
                start_x=self.settings.suptitle_parts_x,
            )

        # Add regular suptitle as text annotation if specified
        if self.settings.suptitle:
            self.ax.text(
                self.settings.suptitle_x,
                self.settings.suptitle_y,
                self.settings.suptitle.format(**vars_settings),
                horizontalalignment=self.settings.suptitle_ha,
                transform=self.ax.transAxes,  # Use axes coordinates (0-1)
                fontsize=self.settings.suptitle_fontsize,
            )

        # Add annotations if specified
        if self.settings.annotations:
            for instance in self.settings.annotations:
                # Convert datetime inputs to float with built-in Matplotlib function
                # This is necessary because matplotlib needs dates as float values
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

                # Add the annotation to the plot
                # Uses special keywords "ANNOT_X" and "ANNOT_Y" for default positions
                self.ax.annotate(
                    str(instance["text"]),
                    (
                        # X position logic: use default + offset, or provided value
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
                        # Y position logic: use default + offset, or provided value
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

        # Hide spines if specified (for a cleaner look)
        if self.settings.hide_spines:
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.spines["bottom"].set_visible(False)
            self.ax.spines["left"].set_visible(False)

        # Hide axes if specified
        if self.settings.hide_axis in ("x", "xaxis", "both", True):
            self.ax.get_xaxis().set_visible(False)
        if self.settings.hide_axis in ("y", "yaxis", "both", True):
            self.ax.get_yaxis().set_visible(False)

        # Hide axis labels if specified
        if self.settings.hide_axis_label in ("x", "xaxis", "both", True):
            self.ax.get_xaxis().get_label().set_visible(False)
        if self.settings.hide_axis_label in ("y", "yaxis", "both", True):
            self.ax.get_yaxis().get_label().set_visible(False)

        # Set legend title if specified
        if self.settings.legend_title:
            self.ax.legend(title=self.settings.legend_title)

        # Hide legend if specified
        if self.settings.hide_legend and self.ax.get_legend():
            self.ax.get_legend().set_visible(False)

        # Adjust layout for optimal spacing
        self.fig.tight_layout()
        return self.fig, self.ax

    def get_figure(self, **kwargs) -> tuple:
        """
        Return the figure, creating it if needed or if settings updates are provided.

        Args:
            **kwargs: Optional settings updates

        Returns:
            tuple: (figure, axes) - The current figure and axes objects
        """
        # Create figure if it doesn't exist yet or if we have new settings
        if (not (self.ax.lines or self.ax.collections)) or len(kwargs) > 0:
            self.create_figure(**kwargs)
        return self.fig, self.ax

    def to_png(
        self,
        fig: Optional[go.Figure | plt.Figure] = None,
        path: Optional[Path] = None,
    ) -> Path:
        """
        Save the figure as a PNG file.

        Args:
            fig: Figure to save (defaults to self.fig if not provided)
            path: Output path (auto-generated if not provided)

        Returns:
            Path: Path to the saved file

        Raises:
            ValueError: If no figure is available to save
        """
        fig = fig or self.fig

        if fig is None:
            raise ValueError(f"No fig to save for {self.settings.file_stem}")

        # Generate default path if not provided
        if not path:
            path = (
                Path(__file__).parent
                / "../../.."
                / self.settings.processed
                / f"{self.settings.file_stem}-{get_now_str()}.png"
            ).resolve()

        # Save based on figure type (plotly or matplotlib)
        if isinstance(fig, go.Figure):
            fig.write_image(path)
        elif isinstance(fig, plt.Figure):
            fig.savefig(path)

        logger.success(f"Successfully saved plot to {path}")

        return path

    def to_html(
        self,
        fig: Optional[go.Figure] = None,
        path: Optional[Path] = None,
    ) -> Path:
        """
        Save the figure as an HTML file (for plotly figures only).

        Args:
            fig: Plotly figure to save (defaults to self.fig if not provided)
            path: Output path (auto-generated if not provided)

        Returns:
            Path: Path to the saved file

        Raises:
            ValueError: If no figure is available or if it's not a plotly figure
        """
        fig = fig or self.fig

        if fig is None:
            raise ValueError(f"No fig to save for {self.settings.file_stem}")

        if not isinstance(fig, go.Figure):
            raise ValueError("Only Plotly objects can be saved to HTML.")

        # Generate default path if not provided
        if not path:
            path = (
                Path(__file__).parent
                / "../../.."
                / self.settings.processed
                / f"{self.settings.file_stem}-{get_now_str()}.html"
            ).resolve()

        fig.write_html(str(path))

        logger.success(f"Successfully saved plot to {path}")

        return path


def is_hyperlink(path: str | Path) -> bool:
    """
    Check if a string is a valid URL/hyperlink.

    Args:
        path: String or Path to check

    Returns:
        bool: True if path is a URL, False otherwise
    """
    # Paths can't be hyperlinks
    if isinstance(path, Path):
        return False

    # Use regex to check for URL pattern
    return bool(
        re.search(
            r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$",
            str(path),
        )
    )


def get_now_str() -> str:
    """
    Get current timestamp as a formatted string.

    Returns:
        str: Current datetime in format 'YYYYMMDD-HHMMSS'
    """
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# Load configuration from TOML file when module is imported
with open((Path(__file__).parent / "../../../config.toml").resolve(), "rb") as f:
    config = tomllib.load(f)
