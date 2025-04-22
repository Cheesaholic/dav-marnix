import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from marn_x.utils.data_transformers import (MessageFileLoader,
                                            author_min_messages,
                                            birthday_congratulations,
                                            calculate_age, create_regex,
                                            get_linear_regression)
# Local imports
from marn_x.utils.settings import AllVars, BasePlot, PlotSettings


class FelicitationSettings(PlotSettings):
    """
    Configuration settings for analyzing birthday congratulation patterns.
    Extends base PlotSettings with specific parameters for this analysis.
    """

    marker_size: int | float  # Size of markers in the scatterplot
    hue_label: str  # Label for the color grouping
    hue_dict: dict  # Dictionary mapping hue values to colors
    min_messages: int  # Minimum number of messages required per author
    birthday_dateformat: str  # Format string for parsing birthday dates
    congratulations_regex: (
        str  # Regular expression pattern to identify congratulation messages
    )
    congratulations_flags: str  # Flags for the regex pattern


class FelicitationLoader(MessageFileLoader):
    """
    Handles loading and processing chat data for birthday congratulation analysis.
    Calculates percentages of congratulations based on age and other factors.
    """

    settings: FelicitationSettings

    def __init__(self, settings: FelicitationSettings):
        """Initialize the loader with settings and process the data."""
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):
        """
        Process raw chat data for birthday congratulation analysis:
        - Filter authors with minimum message count
        - Parse birthday dates
        - Identify congratulation messages
        - Calculate age and congratulation percentages
        - Add hue information for visualization
        """
        # Merge all input DataFrames into one dataset
        self.datafiles.chat = self.datafiles.merge()

        # Filter authors with minimum number of messages
        self.datafiles.chat = author_min_messages(
            self.datafiles.chat,
            self.settings.author_col,
            self.settings.message_col,
            self.settings.min_messages,
        )

        # Convert birthday strings to datetime objects
        birthday_dict = {
            k: pd.to_datetime(v, format=self.settings.birthday_dateformat)
            for k, v in self.datafiles.birthdates.items()
        }

        # Identify congratulation messages using regex pattern
        congratulations_df = birthday_congratulations(
            self.datafiles.chat,
            birthday_dict,
            create_regex(self.settings.congratulations_regex),
        )

        # Calculate age for each author
        congratulations_df[self.settings.xlabel] = [
            calculate_age(birthday_dict[x]) if x in birthday_dict else 0
            for x in congratulations_df[self.settings.author_col]
        ]

        # Calculate percentage of congratulated birthdays
        congratulations_df[self.settings.ylabel] = (
            congratulations_df["congratulated"]
            / (
                congratulations_df["congratulated"]
                + congratulations_df["not_congratulated"]
            )
        ) * 100

        # Add hue information for visualization grouping
        congratulations_df[self.settings.hue_label] = [
            self.datafiles.hue[x] if x in self.datafiles.hue else "?"
            for x in congratulations_df[self.settings.author_col]
        ]

        # Remove entries without x_axis or hue data
        congratulations_df = congratulations_df.loc[
            (congratulations_df[self.settings.xlabel] > 0)
            & (congratulations_df[self.settings.hue_label] != "?")
        ]

        # Map hue labels to colors from settings
        congratulations_df["hue"] = congratulations_df[self.settings.hue_label].map(
            self.settings.hue_dict
        )

        # Store processed data for plotting
        self.datafiles.processed = congratulations_df


class FelicitationPlotter(BasePlot):
    """
    Creates visualizations to show the relationship between age and
    congratulation rate with color coding for different groups.
    """

    settings: FelicitationSettings

    def __init__(self, settings: FelicitationSettings):
        """Initialize with felicitation-specific plot settings."""
        super().__init__(settings)

    def plot(self, data, **kwargs):
        """
        Create a scatterplot visualization showing the relationship between
        age and congratulation percentage, with regression line.

        Args:
            data: Processed DataFrame containing congratulation statistics
            **kwargs: Additional parameters to pass to the figure creation
        """
        # Set up the figure with appropriate dimensions and labels
        super().get_figure(**kwargs)

        # Create scatterplot of age vs. congratulation percentage
        sns.scatterplot(
            x=self.settings.xlabel,
            y=self.settings.ylabel,
            data=data,
            hue=self.settings.hue_label,
            palette=data["hue"].unique().tolist(),
            ec=None,  # No edge color
            ax=self.ax,
            s=self.settings.marker_size,
            legend=False if self.settings.hide_legend else "auto",
        )

        # Add linear regression line to show trend
        get_linear_regression(
            self.ax, data[self.settings.xlabel], data[self.settings.ylabel]
        )

        # Display the plot and save as PNG
        plt.show()
        self.to_png()


def main():
    """
    Main function that orchestrates the birthday congratulation analysis:
    1. Load settings
    2. Process the data
    3. Create visualization with annotation for the outlier
    """
    # Initialize settings from configuration
    settings = FelicitationSettings(**AllVars())

    # Load and process the data
    loader = FelicitationLoader(settings)

    # Create the plotter
    plotter = FelicitationPlotter(settings)

    # Find the outlier with the highest congratulation percentage
    outlier = loader.datafiles.processed[
        loader.datafiles.processed[settings.ylabel]
        == loader.datafiles.processed[settings.ylabel].max()
    ].iloc[0]

    # Create plot with annotation at the outlier point
    plotter.plot(
        loader.datafiles.processed,
        annotation_x=outlier[settings.xlabel],
        annotation_y=outlier[settings.ylabel],
    )


if __name__ == "__main__":
    main()
