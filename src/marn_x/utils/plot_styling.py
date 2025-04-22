import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def continuous_colored_title(
    ax,
    title_parts,
    colors,
    start_x=0.1,
    y_position=1.05,
    x_offset=0.999,
    fontsize=12,
    va="center",
    alpha=0.5,
    pad=2.5,
    edgecolor="none",
    boxstyle="round,pad=0.5",
    weight="normal",
    images=[],
):
    """
    Add a continuous title with different colored backgrounds for each part.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the title to
    title_parts : list of str
        List of title parts to be placed adjacent to each other
    colors : list of str or list of tuples
        List of background colors for each title part
    start_x : float, optional
        Starting horizontal position (0-1) for the first title part, default 0.1
    y_position : float, optional
        Vertical position (in axes coordinates) for all titles, default 1.05
    fontsize : int, optional
        Font size for all title parts, default 12
    va : str, optional
        Vertical alignment, default 'center'
    alpha : float, optional
        Transparency of background boxes, default 0.8
    pad : int, optional
        Padding inside the background boxes, default 3
    edgecolor : str, optional
        Edge color of the background boxes, default 'none'
    boxstyle : str, optional
        Style of the background boxes, default 'round,pad=0.5'

    Returns:
    --------
    list
        List of the text objects created
    """
    text_objects = []
    current_x = start_x

    for unused_i, (part, color) in enumerate(zip(title_parts, colors)):
        if "images" in color:
            image_n = re.findall(r"(?<=images\[)\d(?=\])", color, re.I)[0]
            image = images[int(image_n)]
            color = "none"
        # For all text objects, align left to ensure they start right where we place them
        text = ax.text(
            current_x,
            y_position,
            part,
            transform=ax.transAxes,
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
        renderer = ax.figure.canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        bbox_axes = bbox.transformed(ax.transAxes.inverted())

        if "image" in locals():
            ax.imshow(
                image,
                extent=[bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                alpha=0.1,
                aspect="auto",
            )

        # The next text should start exactly at the end of this one
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
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={"color": color, "ls": "-", "lw": 1, "alpha": alpha},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        x=x,
        y=y,
        data=data,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax,
    )
