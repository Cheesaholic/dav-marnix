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
    file_stem: str
    datafiles: DataFiles
    images: list[Image.Image]
    settings: PlotSettings

    def __init__(self, settings: PlotSettings):
        self.settings = settings

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
        if hasattr(self.settings, "images"):
            self.images = [load_image(image) for image in self.settings.images]
        else:
            raise ValueError(
                f"images in table {self.settings.file_stem} not found in config.toml"
            )

        return self.images

    def clean_transform_data(self):
        pass


def remove_url(text: str) -> str:
    return re.sub(r"^https?:\/\/.*[\r\n]*", "", text)


def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace="")


def remove_image(text: str) -> str:
    return re.sub(r"<Media weggelaten>", "", text)


def remove_more_information(text: str) -> str:
    return re.sub(r"Tik voor meer informatie\.", "", text)


def remove_security_code(text: str) -> str:
    return re.sub(r"Je beveiligingscode voor .* is gewijzigd\.", "", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d*", "", text)


def remove_removed(text: str) -> str:
    return re.sub(r"Dit bericht is verwijderd", "", text)


def remove_edited(text: str) -> str:
    return re.sub(r"<Dit bericht is bewerkt>", "", text)


def remove_exclude_terms(text: str, exclude_terms: list[str] | str = "") -> str:
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
    return df.loc[
        df[author_col].map(df.groupby(author_col)[message_col].count() >= min_messages)
    ]


def fourier_model(timeseries: list | pd.Series, k: int) -> dict:
    # Calculate the number of data points in the timeseries
    T = 1.0
    N = len(timeseries)
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
    if color is None:
        color = "black"

    dim = 1 if x_y == "y" else 0

    # Find the point with the highest/lowest x/y-value

    if highest_lowest == "highest":
        x_coord, y_coord = max(points, key=lambda p: p[dim])
    else:
        x_coord, y_coord = min(points, key=lambda p: p[dim])

    # Label it
    ax.text(
        x_coord + x_offset, y_coord + y_offset, text, ha=ha, va=va, color=color
    ).set_fontweight(fontweight)


def create_fourier_wave(parameters: dict) -> np.ndarray:
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
    b, a = np.polyfit(
        data_x,
        data_y,
        deg=1,
    )

    xseq = np.linspace(
        data_x.min(),
        data_x.max(),
    )

    return ax.plot(xseq, a + b * xseq, color=color, lw=lw, alpha=alpha, label=label)


def get_birthdays(
    birthday: pd.Timestamp, from_date: pd.Timestamp, to_date: pd.Timestamp
) -> pd.DatetimeIndex:

    date_range = pd.date_range(start=from_date, end=to_date)

    day = birthday.day
    month = birthday.month

    birthdays = date_range[(date_range.day == day) & (date_range.month == month)].floor(
        "d"
    )

    return birthdays


def get_now_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_birthday_list(
    author: str, birthday: pd.Timestamp, message_data: pd.DataFrame
) -> pd.DataFrame:

    first_message = message_data.loc[message_data["author"] == author][
        "timestamp"
    ].min()
    last_message = message_data["timestamp"].max()

    birthdays = pd.DataFrame(index=get_birthdays(birthday, first_message, last_message))

    birthdays["author"] = author

    return birthdays


def how_many_congratulations(
    message_data: pd.DataFrame,
    author: str,
    birthdays: pd.DataFrame,
    congratulations_regex: re.Pattern,
) -> tuple:

    first_message = message_data.loc[message_data["author"] == author][
        "timestamp"
    ].min()
    last_message = message_data["timestamp"].max()

    congratulations = (
        message_data.loc[
            (
                message_data["message"].str.match(
                    congratulations_regex.pattern, flags=congratulations_regex.flags
                )
            )
            & (message_data["author"] == author)
        ]["timestamp"]
        .dt.floor("D")
        .to_list()
    )

    if len(birthdays) < 1:
        return 0, 0

    birthdays = birthdays.reset_index(names="timestamp")

    birthdays = birthdays.loc[
        (birthdays["author"] != author)
        & (birthdays["timestamp"] >= first_message)
        & (birthdays["timestamp"] <= last_message)
    ]

    congratulated = len(birthdays.loc[birthdays["timestamp"].isin(congratulations)])

    not_congratulated = len(
        birthdays.loc[~birthdays["timestamp"].isin(congratulations)]
    )

    return congratulated, not_congratulated


def birthday_congratulations(
    message_data: pd.DataFrame, birthday_json: dict, congratulations_regex: re.Pattern
) -> pd.DataFrame:

    authors = message_data["author"].unique()

    birthday_df = pd.DataFrame()

    for author in authors:

        if author not in birthday_json:
            logger.warning(f"No birthday for {author} in birthday-JSON, continuing...")
            continue

        birthday_list = create_birthday_list(
            author, birthday_json[author], message_data
        )

        if birthday_df.empty:
            birthday_df = birthday_list
        else:
            birthday_df = pd.concat([birthday_df, birthday_list])

    birthday_list.sort_index(inplace=True)

    congratulated_df = pd.DataFrame(
        columns=["author", "congratulated", "not_congratulated"]
    )

    for author in authors:
        congratulated, not_congratulated = how_many_congratulations(
            message_data, author, birthday_df, congratulations_regex
        )

        congratulated_df.loc[len(congratulated_df)] = [
            author,
            congratulated,
            not_congratulated,
        ]

    return congratulated_df


def create_regex(regex: str, flags: re.RegexFlag = re.I | re.U) -> re.Pattern:
    return re.compile(regex, flags)


def calculate_age(birthday: pd.Timestamp) -> float:

    return (datetime.now() - birthday) / pd.Timedelta(365.25, "d")
