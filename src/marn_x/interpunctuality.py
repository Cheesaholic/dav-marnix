from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from marn_x.utils.data_transformers import (MessageFileLoader,
                                            create_colored_annotation,
                                            remove_edited, remove_emoji,
                                            remove_image,
                                            remove_more_information,
                                            remove_removed,
                                            remove_security_code, remove_url)
# Local imports
from marn_x.utils.settings import AllVars, BasePlot, PlotSettings


class DistributionSettings(PlotSettings):
    """
    Configuration settings for analyzing punctuation distribution in messages.
    Extends the base PlotSettings with specific parameters for statistical analysis.
    """

    input_settings: dict  # Settings for each input file (colors, distribution type)
    norm_alpha: float | int  # Transparency level for normal distribution curve
    hist_alpha: float | int  # Transparency level for histogram
    interpunction_regex: str  # Regex pattern to identify punctuation marks
    word_regex: str  # Regex pattern to count words
    min_words: int  # Minimum word count for message inclusion
    norm_colnam: str = "ratio_interpunction"  # Column name for punctuation ratio
    p_value_mw: Optional[float] = None  # Mann-Whitney U test p-value
    p_value_ks: Optional[float] = None  # Kolmogorov-Smirnov test p-value


class DistributionLoader(MessageFileLoader):
    """
    Handles loading and processing chat data for punctuation distribution analysis.
    Cleans messages and calculates punctuation-to-word ratios.
    """

    settings: DistributionSettings

    def __init__(self, settings: DistributionSettings):
        """Initialize with settings and process the data."""
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):
        """
        Process raw chat data for punctuation analysis:
        - Clean messages by removing URLs, emojis, etc.
        - Count words and punctuation in each message
        - Calculate punctuation-to-word ratio
        - Filter messages by minimum word count
        """
        # Ensure we have at least two datasets for comparison
        if len(self.datafiles) < 2:
            raise ValueError(
                f"Only found 1 dataset in {self.settings.file_stem}. You need 2 for comparison"
            )

        # Merge all data files into a single dataframe
        self.datafiles.chat = self.datafiles.merge()

        # Clean the message text by removing unwanted elements
        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)
            .apply(remove_image)
            .apply(remove_emoji)
            .apply(remove_more_information)
            .apply(remove_security_code)
            .apply(remove_removed)
            .apply(remove_edited)
            .str.strip()
        )

        # Count words in each message using the word regex
        self.datafiles.chat["n_words"] = self.datafiles.chat[
            self.settings.message_col
        ].str.count(self.settings.word_regex)

        # Filter out messages with fewer than minimum words
        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat["n_words"] >= self.settings.min_words
        ]

        # Count punctuation marks in each message
        self.datafiles.chat["n_interpunction"] = self.datafiles.chat[
            self.settings.message_col
        ].str.count(self.settings.interpunction_regex)

        # Calculate punctuation-to-word ratio
        self.datafiles.chat[self.settings.norm_colnam] = (
            self.datafiles.chat["n_interpunction"] / self.datafiles.chat["n_words"]
        )

        # Store processed data for analysis
        self.datafiles.processed = self.datafiles.chat

    def get_p_values(
        self,
        datafiles: Optional[list[pd.Series | np.ndarray]] = None,
        norm_colnam: Optional[str] = None,
    ) -> tuple:
        """
        Calculate statistical significance between distributions using
        Mann-Whitney U test and Kolmogorov-Smirnov test.

        Args:
            datafiles: Optional list of data series to compare
            norm_colnam: Column name for the normalized punctuation ratio

        Returns:
            Tuple of p-values (Mann-Whitney, Kolmogorov-Smirnov)
        """
        norm_colnam = norm_colnam or self.settings.norm_colnam

        # Use provided datafiles or extract from processed data
        if not datafiles:
            if self.datafiles.processed is not None:
                datafiles = [
                    self.datafiles.processed.loc[
                        self.datafiles.processed["file"] == file
                    ][norm_colnam]
                    for file in self.datafiles.processed["file"].unique()
                ]
            else:
                raise ValueError(
                    f"No datafiles for p_values in {self.settings.file_stem}"
                )
        else:
            # Extract the relevant column if datafiles are DataFrames
            datafiles = [
                (
                    datafile[norm_colnam]
                    if isinstance(datafile, pd.DataFrame)
                    else datafile
                )
                for datafile in datafiles
            ]

        # Check if we have enough data for comparison
        if len(datafiles) < 2:
            logger.warning(
                f"A p-value can't be calculated for {len(datafiles)} distributions. Returning p=1."
            )
            return 1, 1

        if len(datafiles) > 2:
            logger.warning(
                f"A p-value can't be calculated for {len(datafiles)} distributions. Returning p-value for the first 2."
            )

        # Calculate Mann-Whitney U test (non-parametric test for differences in distribution)
        u_stat, p_value_mw = stats.mannwhitneyu(
            datafiles[0], datafiles[1], alternative="two-sided"
        )

        # Calculate Kolmogorov-Smirnov test (test for different distributions)
        ks_stat, p_value_ks = stats.ks_2samp(datafiles[0], datafiles[1])

        return p_value_mw, p_value_ks


class DistributionPlotter(BasePlot):
    """
    Creates visualizations to compare punctuation distribution across datasets.
    Plots histograms and fitted distribution curves.
    """

    settings: DistributionSettings

    def __init__(self, settings: DistributionSettings):
        """Initialize with distribution-specific plot settings."""
        super().__init__(settings)

    def plot(self, data: pd.DataFrame, **kwargs):
        """
        Create a visualization showing the distribution of punctuation usage
        across different datasets with fitted normal distributions.

        Args:
            data: Processed DataFrame containing punctuation statistics
            **kwargs: Additional parameters to pass to the figure creation
        """
        # Set up the figure with appropriate dimensions and labels
        super().get_figure(**kwargs)

        # Determine which files to process
        if "file" not in data.columns:
            files = ["dataset 1"]
        else:
            files = data.file.unique()

        # Process each file and add to the plot
        for file in files:
            # Verify file settings exist
            if file not in self.settings.input_settings:
                raise ValueError(
                    "Message file {file} not in input_settings (config.toml or keyword arguments passed to settings)"
                )

            # Get settings for this specific file
            file_settings = self.settings.input_settings[file]
            datafile = data.loc[data["file"] == file]

            # Create histogram of punctuation ratios
            self.ax.hist(
                datafile[self.settings.norm_colnam],
                density=True,
                alpha=self.settings.hist_alpha,
                color=file_settings["color"],
            )

            # Calculate and plot the fitted normal distribution
            x, pdf = self.get_norm(
                datafile,
                self.settings.norm_colnam,
                file_col=file,
                norm_type=file_settings["type"],
            )

            # Plot the distribution curve
            norm_fig = self.ax.plot(
                x, pdf, file_settings["color"], linewidth=self.settings.linewidth
            )

            # Add sample size annotation
            create_colored_annotation(
                self.ax,
                f"n = {len(datafile)}",
                np.array(norm_fig[0].get_xydata()),
                color=file_settings["color"],
                x_offset=self.settings.annotation_x_offset,
                y_offset=self.settings.annotation_y_offset,
            )

        # Display the plot and save as PNG
        plt.show()
        self.to_png()

    def get_norm(
        self,
        file: pd.DataFrame | pd.Series,
        norm_col: str,
        norm_type: str = "norm",
        file_col: Optional[str] = None,
    ) -> tuple:
        """
        Calculate the normal distribution curve that best fits the data.

        Args:
            file: DataFrame or Series containing the data
            norm_col: Column name for normalized values
            norm_type: Type of distribution to fit ('norm' or 'halfnorm')
            file_col: Name of the file to filter by (if file is DataFrame)

        Returns:
            Tuple of x values and probability density function values
        """
        # Extract the relevant data column
        if isinstance(file, pd.DataFrame):
            if file_col:
                file = file.loc[file["file"] == file_col]

            file = file[norm_col]

        # Create x-axis values for the PDF curve
        x = np.linspace(0, max(file) * 1.2, 1000)

        # Fit appropriate distribution type
        if norm_type == "norm":
            # Fit standard normal distribution
            mu, sigma = stats.norm.fit(file)
            pdf = stats.norm.pdf(x, mu, sigma)

        elif norm_type == "halfnorm":
            # Fit half-normal distribution (for values that can't be negative)
            loc, scale = stats.halfnorm.fit(file)
            pdf = stats.halfnorm.pdf(x, loc=loc, scale=scale)

        else:
            raise ValueError(
                f"norm_type must be norm or halfnorm for file {self.settings.file_stem}..."
            )

        return x, pdf


def main():
    """
    Main function that orchestrates the punctuation distribution analysis:
    1. Load settings
    2. Process the data
    3. Calculate statistical significance
    4. Create visualization with test results
    """
    # Initialize settings from configuration
    settings = DistributionSettings(**AllVars())

    # Load and process the data
    loader = DistributionLoader(settings)

    # Calculate statistical significance between distributions
    p_value_mw, p_value_ks = loader.get_p_values()

    # Create the plotter
    plotter = DistributionPlotter(settings)

    # Create plot with p-values in scientific notation
    plotter.plot(
        loader.datafiles.processed,
        p_value_mw=f"{p_value_mw:.3e}",
        p_value_ks=f"{p_value_ks:.3e}",
    )


if __name__ == "__main__":
    main()
