import re
from datetime import datetime
from typing import Literal, Optional

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D
from PIL import Image
from scipy.fft import fft, fftfreq

from marn_x.utils.file_operations import DataFiles, load_image
from marn_x.utils.settings import PlotSettings


class MessageFileLoader:
    """
    Base class for loading and processing message data files.

    This class handles basic file operations and provides a framework
    for data loading, cleaning, and transformation that can be extended
    by subclasses for specific use cases.

    Attributes:
        file_stem: The base name of the file being processed
        datafiles: Container for data files and processing methods
        images: List of loaded image objects
        settings: Configuration settings for plotting and data processing
    """

    file_stem: str
    datafiles: DataFiles
    images: list[Image.Image]
    settings: PlotSettings

    def __init__(self, settings: PlotSettings):
        """
        Initialize a MessageFileLoader with the given settings.

        Args:
            settings: Configuration settings for file loading and processing
        """
        self.settings = settings

        # Initialize DataFiles with settings from config
        self.datafiles = DataFiles(
            self.settings.input,
            self.settings.raw,
            (
                {}
                if not self.settings.request_headers
                else self.settings.request_headers
            ),
            ([] if not self.settings.json_nests else self.settings.json_nests),
        )

    def get_images(self) -> list[Image.Image]:
        """
        Load images specified in the settings.

        Returns:
            A list of loaded PIL Image objects

        Raises:
            ValueError: If images setting is not found in config
        """
        if hasattr(self.settings, "images"):
            self.images = [load_image(image) for image in self.settings.images]
        else:
            raise ValueError(
                f"images in table {self.settings.file_stem} not found in config.toml"
            )

        return self.images

    def clean_transform_data(self):
        """
        Placeholder method for data cleaning and transformation.

        This method should be overridden by subclasses to implement
        specific data cleaning and transformation logic.
        """
        pass


# Text cleaning functions


def remove_url(text: str) -> str:
    """
    Remove URLs from the provided text.

    Args:
        text: Input text that may contain URLs

    Returns:
        Text with URLs removed
    """
    return re.sub(r"^https?:\/\/.*[\r\n]*", "", text)


def remove_emoji(text: str) -> str:
    """
    Remove emoji characters from the provided text.

    Args:
        text: Input text that may contain emojis

    Returns:
        Text with emojis replaced by empty strings
    """
    return emoji.replace_emoji(text, replace="")


def remove_image(text: str) -> str:
    """
    Remove image placeholders from the provided text.

    Args:
        text: Input text that may contain image placeholders

    Returns:
        Text with image placeholders removed
    """
    return re.sub(r"<Media weggelaten>", "", text)


def remove_more_information(text: str) -> str:
    """
    Remove "Tap for more information" text.

    Args:
        text: Input text that may contain info prompts

    Returns:
        Text with info prompts removed
    """
    return re.sub(r"Tik voor meer informatie\.", "", text)


def remove_security_code(text: str) -> str:
    """
    Remove security code change notifications.

    Args:
        text: Input text that may contain security code notifications

    Returns:
        Text with security code notifications removed
    """
    return re.sub(r"Je beveiligingscode voor .* is gewijzigd\.", "", text)


def remove_numbers(text: str) -> str:
    """
    Remove all numeric digits from the provided text.

    Args:
        text: Input text that may contain numbers

    Returns:
        Text with all digits removed
    """
    return re.sub(r"\d*", "", text)


def remove_removed(text: str) -> str:
    """
    Remove "This message was removed" placeholders.

    Args:
        text: Input text that may contain removal notices

    Returns:
        Text with removal notices removed
    """
    return re.sub(r"Dit bericht is verwijderd", "", text)


def remove_edited(text: str) -> str:
    """
    Remove "This message was edited" placeholders.

    Args:
        text: Input text that may contain edit notices

    Returns:
        Text with edit notices removed
    """
    return re.sub(r"<Dit bericht is bewerkt>", "", text)


def remove_exclude_terms(text: str, exclude_terms: list[str] | str = "") -> str:
    """
    Remove specified terms from the provided text.

    Args:
        text: Input text from which terms should be removed
        exclude_terms: Either a regex pattern string or a list of terms to exclude

    Returns:
        Text with specified terms removed

    Raises:
        ValueError: If exclude_terms is neither a string nor a list
    """
    if isinstance(exclude_terms, str):
        regex = exclude_terms
    elif isinstance(exclude_terms, list):
        regex = r"\b(" + "|".join(exclude_terms) + r")\b"
    else:
        raise ValueError("Input must be str or list of str")
    return re.sub(regex, "", text, flags=re.IGNORECASE)


def author_min_messages(
    df: pd.DataFrame, author_col: str, message_col: str, min_messages: int
) -> pd.DataFrame:
    """
    Filter DataFrame to only include authors with at least min_messages.

    Args:
        df: DataFrame containing message data
        author_col: Column name containing author information
        message_col: Column name containing message content
        min_messages: Minimum number of messages required for inclusion

    Returns:
        Filtered DataFrame containing only authors with sufficient messages
    """
    return df.loc[
        df[author_col].map(df.groupby(author_col)[message_col].count() >= min_messages)
    ]


def fourier_model(timeseries: list | pd.Series, k: int) -> dict:
    """
    Create a Fourier model with the k largest frequency components.

    This function performs a Fast Fourier Transform (FFT) on the given timeseries
    and extracts the k frequency components with the largest amplitudes.

    Args:
        timeseries: Input time series data
        k: Number of frequency components to extract

    Returns:
        Dictionary containing:
            - x: Time vector
            - frequencies: The k most significant frequencies
            - amplitudes: Amplitudes of the k frequencies
            - phases: Phases of the k frequencies
    """
    # Calculate the number of data points in the timeseries
    T = 1.0  # Sampling interval
    N = len(timeseries)  # Number of data points

    # Generate a time vector 'x' from 0 to N*T (excluding the endpoint) evenly spaced
    x = np.linspace(0.0, N * T, N, endpoint=False)

    # Perform the Fourier Transform of the timeseries
    yf = fft(timeseries)

    # Generate the frequency bins for the first half of the Fourier Transform result
    # This represents the positive frequencies up to the Nyquist frequency
    xf = fftfreq(N, T)[: N // 2]

    # Identify indices of the 'k' largest frequencies by their magnitude in the first half of the Fourier spectrum
    indices = np.argsort(np.abs(yf[0 : N // 2]))[-k:]

    # Extract the frequencies corresponding to the 'k' largest magnitudes
    frequencies = xf[indices]

    # Calculate the amplitudes of these frequencies as twice the magnitude divided by N
    # This accounts for the symmetry of the Fourier Transform for real signals
    amplitudes = 2.0 / N * np.abs(yf[indices])

    # Extract the phases of these frequencies and adjust by adding pi/2 to align phases
    phases = np.angle(yf[indices]) + 1 / 2 * np.pi

    # Return a dictionary of the model parameters: 'x', 'frequencies', 'amplitudes', 'phases'
    return {
        "x": x,
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "phases": phases,
    }


def create_colored_annotation(
    ax: plt.Axes,
    text: str,
    points: np.ndarray | np.ma.MaskedArray,
    color: Optional[str | int | float] = None,
    highest_lowest: Literal["highest", "lowest"] = "highest",
    x_y: Literal["x", "y"] = "y",
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    ha: str = "center",
    va: str = "center",
    fontweight: str = "bold",
):
    """
    Add a text annotation to a matplotlib plot at the highest or lowest point.

    Args:
        ax: Matplotlib Axes object to add annotation to
        text: Annotation text to display
        points: Array of (x,y) points to search for highest/lowest
        color: Color of the annotation text (default: black)
        highest_lowest: Whether to annotate the highest or lowest point
        x_y: Dimension to determine highest/lowest (x or y)
        x_offset: Horizontal offset for the annotation
        y_offset: Vertical offset for the annotation
        ha: Horizontal alignment of text
        va: Vertical alignment of text
        fontweight: Font weight for the text
    """
    if color is None:
        color = "black"

    # Determine which dimension to use for finding min/max
    dim = 1 if x_y == "y" else 0

    # Find the point with the highest/lowest x/y-value
    if highest_lowest == "highest":
        x_coord, y_coord = max(points, key=lambda p: p[dim])
    else:
        x_coord, y_coord = min(points, key=lambda p: p[dim])

    # Add text annotation with specified formatting
    ax.text(
        x_coord + x_offset, y_coord + y_offset, text, ha=ha, va=va, color=color
    ).set_fontweight(fontweight)


def create_fourier_wave(parameters: dict) -> np.ndarray:
    """
    Generate a waveform from Fourier parameters.

    Takes the output of fourier_model() and reconstructs the time series
    using the extracted frequency components.

    Args:
        parameters: Dictionary containing 'x', 'frequencies', 'amplitudes', and 'phases'

    Returns:
        Reconstructed time series as a numpy array
    """
    # Extract the time vector 'x' from the parameters
    x = parameters["x"]

    # Extract the frequencies, amplitudes, and phases from the parameters
    frequencies = parameters["frequencies"]
    amplitudes = parameters["amplitudes"]
    phases = parameters["phases"]

    # Initialize a zero array 'y' of the same shape as 'x' to store the model output
    y = np.zeros_like(x)

    # Add each sine wave component to 'y' based on the extracted frequencies, amplitudes, and phases
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        y += amp * np.sin(2.0 * np.pi * freq * x + phase)

    # Return the composite signal 'y' as the sum of the sine wave components
    return y


def get_linear_regression(
    ax: plt.Axes,
    data_x: pd.Series,
    data_y: pd.Series,
    color="k",
    lw=1,
    alpha=0.5,
    label="LSRL",
) -> list[Line2D]:
    """
    Add a linear regression line to a matplotlib plot.

    Calculates and plots the least squares regression line for given x and y data.

    Args:
        ax: Matplotlib Axes object to add the line to
        data_x: X-axis data series
        data_y: Y-axis data series
        color: Line color (default: black)
        lw: Line width (default: 1)
        alpha: Line transparency (default: 0.5)
        label: Line label for legend (default: "LSRL")

    Returns:
        List of Line2D objects representing the regression line
    """
    # Calculate linear regression coefficients (slope and intercept)
    b, a = np.polyfit(
        data_x,
        data_y,
        deg=1,
    )

    # Create a sequence of x values spanning the range of data_x
    xseq = np.linspace(
        data_x.min(),
        data_x.max(),
    )

    # Plot the regression line and return the Line2D objects
    return ax.plot(xseq, a + b * xseq, color=color, lw=lw, alpha=alpha, label=label)


def get_birthdays(
    birthday: pd.Timestamp, from_date: pd.Timestamp, to_date: pd.Timestamp
) -> pd.DatetimeIndex:
    """
    Get all birthdays between two dates.

    Args:
        birthday: Original birthday timestamp
        from_date: Start date for the range
        to_date: End date for the range

    Returns:
        DatetimeIndex containing all birthday occurrences in the date range
    """
    # Create a date range spanning from_date to to_date
    date_range = pd.date_range(start=from_date, end=to_date)

    # Extract day and month from the birthday
    day = birthday.day
    month = birthday.month

    # Filter the date range to include only dates with matching day and month
    # Floor to day to remove any time component
    birthdays = date_range[(date_range.day == day) & (date_range.month == month)].floor(
        "d"
    )

    return birthdays


def create_birthday_list(
    author: str, birthday: pd.Timestamp, message_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a DataFrame of an author's birthdays within the message history timeframe.

    Args:
        author: Name of the person whose birthdays to track
        birthday: Original birthday timestamp
        message_data: DataFrame containing message history with timestamps

    Returns:
        DataFrame with timestamps of all birthdays for the author in the date range
    """
    # Find the first message from this author and the last message in the dataset
    first_message = message_data.loc[message_data["author"] == author][
        "timestamp"
    ].min()
    last_message = message_data["timestamp"].max()

    # Get all birthdays in the date range
    birthdays = pd.DataFrame(index=get_birthdays(birthday, first_message, last_message))

    # Add author column to the birthdays DataFrame
    birthdays["author"] = author

    return birthdays


def how_many_congratulations(
    message_data: pd.DataFrame,
    author: str,
    birthdays: pd.DataFrame,
    congratulations_regex: re.Pattern,
) -> tuple:
    """
    Count how many birthdays an author has acknowledged with congratulations.

    Args:
        message_data: DataFrame containing message history
        author: Name of the person sending congratulations
        birthdays: DataFrame of birthdays for all authors
        congratulations_regex: Regex pattern to identify congratulation messages

    Returns:
        Tuple of (number_of_birthdays_congratulated, number_of_birthdays_not_congratulated)
    """
    # Find the first message from this author and the last message in the dataset
    first_message = message_data.loc[message_data["author"] == author][
        "timestamp"
    ].min()
    last_message = message_data["timestamp"].max()

    # Get all dates when this author sent congratulation messages
    congratulations = (
        message_data.loc[
            (
                message_data["message"].str.match(
                    congratulations_regex.pattern, flags=congratulations_regex.flags
                )
            )
            & (message_data["author"] == author)
        ]["timestamp"]
        .dt.floor("D")  # Floor to day to ignore time
        .to_list()
    )

    # Return zeros if no birthdays in the dataset
    if len(birthdays) < 1:
        return 0, 0

    # Reset index to make timestamp a column
    birthdays = birthdays.reset_index(names="timestamp")

    # Filter birthdays to exclude the author's own birthday and restrict to message timeframe
    birthdays = birthdays.loc[
        (birthdays["author"] != author)
        & (birthdays["timestamp"] >= first_message)
        & (birthdays["timestamp"] <= last_message)
    ]

    # Count birthdays that were congratulated (dates match)
    congratulated = len(birthdays.loc[birthdays["timestamp"].isin(congratulations)])

    # Count birthdays that were not congratulated
    not_congratulated = len(
        birthdays.loc[~birthdays["timestamp"].isin(congratulations)]
    )

    return congratulated, not_congratulated


def birthday_congratulations(
    message_data: pd.DataFrame, birthday_json: dict, congratulations_regex: re.Pattern
) -> pd.DataFrame:
    """
    Analyze birthday congratulations patterns across all authors.

    Args:
        message_data: DataFrame containing message history
        birthday_json: Dictionary mapping author names to birthday timestamps
        congratulations_regex: Regex pattern to identify congratulation messages

    Returns:
        DataFrame with statistics on congratulated vs. not congratulated birthdays by author
    """
    # Get list of all unique authors in the message data
    authors = message_data["author"].unique()

    # Initialize empty DataFrame to hold all birthdays
    birthday_df = pd.DataFrame()

    # Process each author
    for author in authors:
        # Skip authors with missing birthday information
        if author not in birthday_json:
            logger.warning(f"No birthday for {author} in birthday-JSON, continuing...")
            continue

        # Create list of birthdays for this author
        birthday_list = create_birthday_list(
            author, birthday_json[author], message_data
        )

        # Add to master birthday DataFrame
        if birthday_df.empty:
            birthday_df = birthday_list
        else:
            birthday_df = pd.concat([birthday_df, birthday_list])

    # Sort birthdays by date
    birthday_list.sort_index(inplace=True)

    # Initialize results DataFrame
    congratulated_df = pd.DataFrame(
        columns=["author", "congratulated", "not_congratulated"]
    )

    # Calculate congratulation statistics for each author
    for author in authors:
        congratulated, not_congratulated = how_many_congratulations(
            message_data, author, birthday_df, congratulations_regex
        )

        # Add results to the output DataFrame
        congratulated_df.loc[len(congratulated_df)] = [
            author,
            congratulated,
            not_congratulated,
        ]

    return congratulated_df


def create_regex(regex: str, flags: re.RegexFlag = re.I | re.U) -> re.Pattern:
    """
    Create a compiled regex pattern with specified flags.

    Args:
        regex: Regular expression pattern string
        flags: Regex flags (default: case-insensitive and Unicode)

    Returns:
        Compiled regex pattern
    """
    return re.compile(regex, flags)


def calculate_age(birthday: pd.Timestamp) -> float:
    """
    Calculate age in years from a birthday.

    Args:
        birthday: Birthday timestamp

    Returns:
        Age in years as a float
    """
    # Calculate difference between now and birthday, convert to years
    return (datetime.now() - birthday) / pd.Timedelta(365.25, "d")
