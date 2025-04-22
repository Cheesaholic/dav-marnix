from typing import Literal

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import seaborn as sns

from marn_x.utils.data_transformers import (MessageFileLoader,
                                            create_colored_annotation,
                                            remove_emoji, remove_image,
                                            remove_numbers, remove_url)
from marn_x.utils.plot_styling import stripplot_mean_line
# Local imports
from marn_x.utils.settings import AllVars, BasePlot, PlotSettings


class PeriodSettings(PlotSettings):
    """
    Configuration settings for period analysis visualization.
    Extends the base PlotSettings with specific parameters for this analysis.
    """

    markers: list[str]  # List of marker styles for the plot
    marker_size: int | float  # Size of markers in the plot
    min_tokens: int  # Minimum number of tokens required for message processing
    min_long_messages: int  # Minimum number of messages required per file/author
    end_of_sentence_list: list[str]  # List of characters that signify end of sentence
    annotation_location_highest_lowest: Literal[
        "higest", "lowest"
    ]  # Position for annotations
    annotation_location_x_y: Literal["x", "y"]  # Axis for annotation positioning
    mean_color: str  # Color for the mean line
    mean_alpha: float | int  # Transparency for the mean line


class PeriodLoader(MessageFileLoader):
    """
    Handles loading and processing of chat data for period usage analysis.
    Calculates the percentage of messages ending with specified punctuation.
    """

    settings: PeriodSettings

    def __init__(self, settings: PeriodSettings):
        """Initialize the loader with appropriate settings and process the data."""
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):
        """
        Process raw chat data to prepare for analysis:
        - Clean messages by removing URLs, images, emojis, numbers
        - Track which messages end with periods or other sentence-ending punctuation
        - Calculate percentage statistics for visualization
        """
        # Merge all data files into a single dataframe
        self.datafiles.chat = self.datafiles.merge(capitalize_filename=True)

        # Clean the message text by removing unwanted elements
        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)
            .apply(remove_image)
            .apply(remove_emoji)
            .apply(remove_numbers)
            .str.strip()
        )

        # Check if each message ends with a sentence-ending character
        self.datafiles.chat["endswith_period"] = self.datafiles.chat[
            self.settings.message_col
        ].str.endswith(tuple(self.settings.end_of_sentence_list))

        # Filter out messages that are too short
        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat[self.settings.message_col].str.count(" ") + 1
            > self.settings.min_tokens
        ]

        # Group data by file, author, and whether it ends with a period
        df = self.datafiles.chat.groupby(["file", "author", "endswith_period"]).count()[
            ["timestamp"]
        ]

        # Count total messages per file and author
        df["cnt_long_messages"] = df.groupby(["file", "author"])["timestamp"].transform(
            "sum"
        )

        # Calculate the percentage of messages ending with period for each group
        df["pct_endswith_period"] = (df["timestamp"] / df["cnt_long_messages"]) * 100

        # Filter to only include rows where endswith_period is True
        result = df[df.index.get_level_values("endswith_period")]

        # Clean up the results dataframe
        result = result.drop(columns=["timestamp"])
        result.reset_index(level=["endswith_period", "author"], drop=True, inplace=True)
        result.reset_index(inplace=True)

        # Store processed data for plotting
        self.datafiles.processed = result


class PeriodPlotter(BasePlot):
    """
    Creates visualizations to show the distribution of period usage across files.
    Uses stripplots with custom markers and annotations.
    """

    settings: PeriodSettings

    def __init__(self, settings: PeriodSettings):
        """Initialize with period-specific plot settings"""
        super().__init__(settings)

    def plot(self, data, **kwargs):
        """
        Create a visualization showing the percentage of messages ending with
        sentence-ending punctuation for each file.

        Args:
            data: Processed DataFrame containing period usage statistics
            **kwargs: Additional parameters to pass to the figure creation
        """
        # Set up the figure with title mentioning which end-of-sentence markers were used
        super().get_figure(
            end_of_sentence_list=" ".join(self.settings.end_of_sentence_list), **kwargs
        )

        # Filter out files with too few messages
        data = data.loc[data["cnt_long_messages"] >= self.settings.min_long_messages]

        # Create the stripplot showing distribution of period usage
        sns.stripplot(
            data=data,
            x="pct_endswith_period",
            y="file",
            hue="file",
            size=self.settings.marker_size,
            jitter=True,
            ax=self.ax,
        )

        # Add mean lines to show average values
        stripplot_mean_line(
            self.ax,
            data,
            "pct_endswith_period",
            "file",
            color=self.settings.mean_color,
            alpha=self.settings.mean_alpha,
        )

        # Get the y-axis labels (file names)
        labels = [x.get_text() for x in self.ax.get_yticklabels()]

        # Customize each data point collection (one per file)
        for collection_num in range(len(self.ax.collections)):
            # Add annotations for each filename at the highest/lowest value
            create_colored_annotation(
                self.ax,
                labels[collection_num],
                self.ax.collections[collection_num].get_offsets(),
                highest_lowest=self.settings.annotation_location_highest_lowest,
                x_y=self.settings.annotation_location_x_y,
                color=self.ax.collections[collection_num].get_facecolor()[0],
                x_offset=self.settings.annotation_x_offset,
                y_offset=self.settings.annotation_y_offset,
                fontweight=self.settings.annotation_fontweight,
            )

            # Apply custom marker shapes to each collection, to help colorblind people read the plot
            marker_obj = mmarkers.MarkerStyle(self.settings.markers[collection_num])
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            self.ax.collections[collection_num].set_paths(
                [path] * len(self.ax.collections[collection_num].get_offsets())
            )

        # Display the plot and save as PNG
        plt.show()
        self.to_png()


def main():
    """
    Main function that orchestrates the period analysis process:
    1. Load settings
    2. Process the data
    3. Create and display the visualization
    """
    # Initialize settings from configuration
    settings = PeriodSettings(**AllVars())

    # Load and process the data
    loader = PeriodLoader(settings)

    # Create the plotter and generate visualization
    plotter = PeriodPlotter(settings)
    plotter.plot(loader.datafiles.processed)


if __name__ == "__main__":
    main()
