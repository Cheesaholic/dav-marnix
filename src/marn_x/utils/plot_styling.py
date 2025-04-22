"""
plot_styling.py - Utilities for enhanced and customized matplotlib plot styling

This module provides functions to create visually appealing and informative plot elements
such as multi-colored titles and specialized statistical visualizations.

Functions:
    continuous_colored_title: Creates a sequence of title segments with different background colors
    stripplot_mean_line: Adds a mean line to a stripplot for better visual interpretation
"""

import re
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg


def continuous_colored_title(
    ax: plt.Axes,
    title_parts: list,
    colors: list,
    start_x: float = 0.1,
    y_position: float = 1.05,
    x_offset: float = 0.999,
    fontsize: int = 12,
    va: str = "center",
    alpha: float = 0.5,
    pad: float = 2.5,
    edgecolor: str = "none",
    boxstyle: str = "round,pad=0.5",
    weight: str = "normal",
    images: list = [],
) -> list:
    """
    Add a continuous title with different colored backgrounds for each part.

    Creates a sequence of adjacent title segments with custom background colors.
    Optionally supports overlaying images on text backgrounds, which can be useful
    for adding visual indicators or branding to plot titles.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the title to
    title_parts : list of str
        List of title parts to be placed adjacent to each other
    colors : list of str or list of tuples
        List of background colors for each title part (can include 'images[n]' references)
        Colors can be any valid matplotlib color specification (name, hex, RGB, etc.)
    start_x : float, optional
        Starting horizontal position (0-1) for the first title part, default 0.1
    y_position : float, optional
        Vertical position (in axes coordinates) for all titles, default 1.05
    x_offset : float, optional
        Multiplier for horizontal positioning between segments (adjust for spacing), default 0.999
        Values slightly less than 1 create tighter spacing between segments
    fontsize : int, optional
        Font size for all title parts, default 12
    va : str, optional
        Vertical alignment, default 'center'
    alpha : float, optional
        Transparency of background boxes, default 0.5
    pad : float, optional
        Padding inside the background boxes, default 2.5
    edgecolor : str, optional
        Edge color of the background boxes, default 'none'
    boxstyle : str, optional
        Style of the background boxes, default 'round,pad=0.5'
        See matplotlib.patches.BoxStyle for available styles
    weight : str, optional
        Font weight for text, default 'normal'
        Options include: 'normal', 'bold', 'light', etc.
    images : list, optional
        List of image objects that can be overlaid on text backgrounds, default empty list
        Images should be compatible with matplotlib's imshow function

    Returns:
    --------
    list
        List of the text objects created for further manipulation if needed

    Example:
    --------
    >>> fig, ax = plt.subplots()
    >>> title_parts = ["Part 1", "Part 2", "Part 3"]
    >>> colors = ["lightblue", "lightgreen", "lightyellow"]
    >>> text_objects = continuous_colored_title(ax, title_parts, colors)
    """
    text_objects = []
    current_x = start_x

    for unused_i, (part, color) in enumerate(zip(title_parts, colors)):
        # Check if this segment should have an image overlay
        if isinstance(color, str) and "images" in color:
            # Extract the image index from the color string (e.g., "images[0]" → 0)
            image_n = re.findall(r"(?<=images\[)\d(?=\])", color, re.I)[0]
            image = images[int(image_n)]
            color = "none"  # Make background transparent when using image

        # Create text object with custom styling and position
        # For all text objects, align left to ensure they start right where we place them
        text = ax.text(
            current_x,
            y_position,
            part,
            transform=ax.transAxes,  # Use axes coordinates (0-1 range)
            fontsize=fontsize,
            weight=weight,
            ha="left",
            va=va,
            bbox=dict(
                boxstyle=boxstyle,
                facecolor=color,
                alpha=alpha,
                pad=pad,
                edgecolor=edgecolor,
            ),
        )
        text_objects.append(text)

        # Calculate the exact end position of this text box to place the next one
        # We need to get the actual renderer to compute the text dimensions
        canvas = cast(FigureCanvasAgg, ax.figure.canvas)
        renderer = canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        bbox_axes = bbox.transformed(ax.transAxes.inverted())

        # If this segment has an image, overlay it on the text background
        if "image" in locals():
            # Position image to match the text bounding box
            ax.imshow(
                image,
                extent=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                alpha=0.1,  # Low alpha for subtle image effect
                aspect="auto",
            )
            # Clean up local variable to avoid affecting the next iteration
            del image

        # Calculate position for the next text segment
        # The next text should start exactly at the end of this one
        # x_offset allows fine-tuning of segment spacing (values < 1 create tighter spacing)
        current_x = bbox_axes.x1 * x_offset

    return text_objects


def stripplot_mean_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str = "k",
    alpha: float = 1.0,
):
    """
    Add a mean line to a stripplot using a customized boxplot.

    This function creates a boxplot with only the mean line visible,
    which can be overlaid on a stripplot to show the central tendency.
    The implementation uses seaborn's boxplot with most elements hidden
    except for the mean line.

    Parameters:
    -----------
    ax : plt.Axes
        The matplotlib axes object to draw on
    data : pd.DataFrame
        DataFrame containing the data to plot
    x : str
        Column name in data for the x-axis categories
    y : str
        Column name in data for the y-axis values
    color : str, optional
        Color of the mean line, default 'k' (black)
    alpha : float, optional
        Transparency of the mean line, default 1.0 (opaque)

    Returns:
    --------
    None
        The function modifies the provided axes object in-place

    Notes:
    ------
    This function is typically used after creating a stripplot on the same axes,
    to add a visual reference for the mean value of each category.

    Example:
    --------
    >>> fig, ax = plt.subplots()
    >>> sns.stripplot(x='category', y='value', data=df, ax=ax)
    >>> stripplot_mean_line(ax, df, 'category', 'value', color='red')
    """
    # Create a boxplot but only show the mean line
    # This is a clever way to overlay just the mean on a stripplot
    sns.boxplot(
        showmeans=True,  # Show the mean as a line
        meanline=True,  # Use a line instead of a point for the mean
        meanprops={
            "color": color,
            "ls": "-",
            "lw": 1,
            "alpha": alpha,
        },  # Mean line properties
        medianprops={"visible": False},  # Hide the median line
        whiskerprops={"visible": False},  # Hide the whiskers
        zorder=10,  # Ensure the mean line appears above other plot elements
        x=x,
        y=y,
        data=data,
        showfliers=False,  # Hide outlier points
        showbox=False,  # Hide the box
        showcaps=False,  # Hide the caps on the whiskers
        ax=ax,
    )
