from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local import
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, create_colored_annotation,
                             create_fourier_wave, fourier_model)


class WorstenbroodjesSettings(PlotSettings):
    """
    Configuration settings for analyzing worstenbroodjes data (Dutch sausage rolls).
    Extends the base PlotSettings with specific parameters for time series and Fourier analysis.
    """

    api_data_col: str  # Column name for the data from API
    api_dt_format: str  # Datetime format string for parsing API timestamps
    holidays: list[dict]  # List of holiday periods to highlight on the plot
    aggregation_frequency: (
        str  # Frequency for time series aggregation (e.g., 'W' for weekly)
    )
    fourier_components: int  # Number of Fourier components to use in the model
    fourier_color: str  # Color for the Fourier model line
    fourier_alpha: float  # Transparency for the Fourier model line
    fourier_col: str  # Column name for storing Fourier predictions
    data_color: str  # Color for the actual data line


class WorstenbroodjesLoader(MessageFileLoader):
    """
    Handles loading and processing Wikipedia API data related to worstenbroodjes.
    Performs time series aggregation and Fourier analysis.
    """

    settings: WorstenbroodjesSettings

    def __init__(self, settings: WorstenbroodjesSettings):
        """Initialize with worstenbroodjes-specific settings"""
        super().__init__(settings)

    def clean_transform_data(self):
        """
        Process raw Wikipedia API data for time series analysis:
        - Convert timestamps to datetime objects
        - Aggregate data by specified frequency
        - Normalize data for Fourier analysis
        """
        # Get the Wikipedia API data
        df = self.datafiles.merge()

        # Convert timestamp strings to datetime objects
        df[self.settings.timestamp_col] = pd.to_datetime(
            df[self.settings.timestamp_col], format=self.settings.api_dt_format
        )

        # Aggregate data by the specified frequency (e.g., daily, weekly)
        agg = (
            df.groupby(
                [
                    pd.Grouper(
                        key=self.settings.timestamp_col,
                        freq=self.settings.aggregation_frequency,
                    )
                ]
            )[self.settings.api_data_col]
            .sum()  # Sum values within each time period
            .reset_index()
            .sort_values(self.settings.timestamp_col)
            .set_index(
                self.settings.timestamp_col
            )  # Use timestamp as index for plotting
        )

        # Convert data to float type
        agg[self.settings.api_data_col] = agg[self.settings.api_data_col].astype(float)
        # Create normalized version (mean-centered) for Fourier analysis
        agg[self.settings.api_data_col + "_nrm"] = (
            agg[self.settings.api_data_col] - agg[self.settings.api_data_col].mean()
        )

        # Store processed data
        self.datafiles.processed = agg

    def create_fourier_model(
        self,
        custom_df: Optional[pd.DataFrame] = None,
        data_col: Optional[str] = None,
        fourier_components: Optional[int] = None,
        fourier_col: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create a Fourier model to identify cyclical patterns in the time series data.

        Args:
            custom_df: Optional custom DataFrame to use instead of processed data
            data_col: Column name for the data to model
            fourier_components: Number of Fourier components to use
            fourier_col: Column name to store the Fourier model predictions

        Returns:
            NumPy array of Fourier model predictions
        """
        # Use provided parameters or defaults from settings
        data = custom_df or self.datafiles.processed
        data_col = data_col or self.settings.api_data_col
        fourier_components = fourier_components or self.settings.fourier_components
        fourier_col = fourier_col or self.settings.fourier_col

        # Ensure we have data to model
        if data is None:
            raise ValueError(
                f"No DataFrame to create fourier model for {self.settings.file_stem}"
            )

        # Fit Fourier model to the normalized data
        parameters = fourier_model(data[data_col + "_nrm"], fourier_components)

        # Generate predictions from the model parameters
        y = create_fourier_wave(parameters)

        # Add back the mean to denormalize predictions
        y += data[data_col].mean()

        # Store predictions in the processed dataframe if using default data
        if not custom_df and self.datafiles.processed is not None:
            self.datafiles.processed[fourier_col] = y

        return y


class WorstenbroodjesPlotter(BasePlot):
    """
    Creates visualizations of worstenbroodjes time series data with Fourier model overlay
    and holiday period highlighting.
    """

    settings: WorstenbroodjesSettings

    def __init__(self, settings: WorstenbroodjesSettings):
        """Initialize with worstenbroodjes-specific plot settings"""
        super().__init__(settings)

    def plot(self, data, **kwargs):
        """
        Create a time series visualization showing worstenbroodjes data with Fourier model
        and highlighted holiday periods.

        Args:
            data: Processed DataFrame containing time series and Fourier predictions
            **kwargs: Additional parameters to pass to the figure creation
        """
        # Set up the figure with appropriate dimensions and labels
        super().get_figure(**kwargs)

        # Plot the actual data values
        ann_plt = self.ax.plot(
            data.index,  # x-axis: timestamps
            data[self.settings.api_data_col],  # y-axis: data values
            linewidth=self.settings.linewidth,
            color=self.settings.data_color,
        )

        # Plot the Fourier model predictions as a dashed line
        self.ax.plot(
            data.index,
            data[self.settings.fourier_col],
            linestyle="--",
            linewidth=self.settings.linewidth,
            color=self.settings.fourier_color,
            alpha=self.settings.fourier_alpha,
        )

        # Highlight holiday periods as colored spans on the plot
        for instance in self.settings.holidays:
            self.ax.axvspan(
                instance["start"],  # Start date of holiday
                instance["end"],  # End date of holiday
                facecolor=instance["color"],  # Background color for holiday
                alpha=instance["alpha"],  # Transparency for holiday highlight
            )

        # Add annotation label for the data line
        create_colored_annotation(
            self.ax,
            self.settings.ylabel,  # Label text (from settings)
            ann_plt[0].get_xydata(),  # Position based on data points
            x_y="x",  # Position along x-axis
            color=self.settings.data_color,
            x_offset=self.settings.annotation_x_offset,
            y_offset=self.settings.annotation_y_offset,
            fontweight=self.settings.annotation_fontweight,
            ha="left",  # Left-align the text
        )

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()
        self.to_png()  # Save the plot as PNG file


def main():
    """
    Main function that orchestrates the worstenbroodjes analysis:
    1. Load settings
    2. Process the Wikipedia API data
    3. Create Fourier model to identify patterns
    4. Create visualization with model overlay and holiday highlights
    """
    # Initialize settings from configuration
    settings = WorstenbroodjesSettings(**AllVars())

    # Load and process the data
    loader = WorstenbroodjesLoader(settings)
    loader.clean_transform_data()
    loader.create_fourier_model()

    # Create the plotter
    plotter = WorstenbroodjesPlotter(settings)

    # Create plot with annotation at the maximum data value
    plotter.plot(
        loader.datafiles.processed,
        annotation_y=loader.datafiles.processed[settings.api_data_col].max(),
    )


if __name__ == "__main__":
    main()
