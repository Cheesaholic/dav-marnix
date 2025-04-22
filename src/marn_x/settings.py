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


class DataFiles:
    message_file_paths: dict[str, Path | str]
    datafiles: dict[str, str | pd.DataFrame]
    request_headers: dict[str, str]
    json_nests: list[str]
    images: list[Image.Image]

    chat: Optional[pd.DataFrame] = None
    processed: Optional[pd.DataFrame] = None
    topic_model: Optional[pd.DataFrame] = None

    def __init__(
        self,
        input: dict[str, str],
        raw: str,
        request_headers: dict,
        json_nests: list[str],
    ):
        self.input_path = (Path(__file__).parent / "../.." / raw).resolve()

        self.message_file_paths = {
            k: (
                (self.input_path / Path(message_file)).resolve()
                if not is_hyperlink(message_file)
                else message_file
            )
            for (k, message_file) in input.items()
        }

        self.datafiles = {
            k: load_dataframe(
                path,
                self,
                request_headers=request_headers,
                json_nests=json_nests,
            )
            for (k, path) in self.message_file_paths.items()
        }

        for key in self.datafiles:
            logger.info(f"Loaded dataframe from input.{key}")
            setattr(self, key, self.datafiles[key])

    def __iter__(self):
        return self.datafiles

    def __len__(self) -> int:
        return len(self.datafiles)

    def parse_json(self, json_file: dict, json_nests: list) -> dict:
        """

        Gets nested level belonging to the JSON file from config.toml.
        Uses self.file_stem as suffix.
        Takes and returns JSON. If no config is present, returns same JSON.

        """

        for level in json_nests:
            json_file = json_file[level]

        return json_file

    def all(self, values: bool = False) -> Iterable | list:
        if values:
            return list(self.datafiles.values())
        else:
            return self.datafiles.items()

    def merge(
        self,
        files: Optional[list[pd.DataFrame]] = None,
        capitalize_filename: bool = False,
    ) -> pd.DataFrame:
        if not files:
            files = []
            for file_name, datafile in self.all():
                if isinstance(datafile, pd.DataFrame):
                    if capitalize_filename:
                        file_name = file_name.capitalize()
                    datafile["file"] = file_name
                    files.append(datafile)

        merge = pd.concat(files, axis=0)

        if len(files) > 1:
            logger.info(f"Merged {len(files)} dataframes.")

        return merge


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
                / "../.."
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
                / "../.."
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


def get_now_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


def get_api_data(endpoint: str, headers: dict = {}, payload: dict = {}) -> Response:
    """Connects to API via endpoint and variables in string.
    Returns JSON-object."""

    logger.info(
        f"API Call. Endpoint: {endpoint}, headers: {headers}, payload: {payload}"
    )

    try:
        # Fire response. Throw error if request takes longer than 20 seconds
        response = get(endpoint, headers=headers, data=payload, timeout=20)

        logger.info(f"API HTML response-code {response.status_code}")

        # Raises HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status()

    except HTTPError as http_err:
        logger.error(
            f"HTTP error occurred: {http_err}"
        )  # e.g., 404 Not Found, 500 Internal Server Error
    except ConnectionError as conn_err:
        logger.error(
            f"Connection error occurred: {conn_err}"
        )  # Issues with network connectivity
    except Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err}")  # Request timed out
    except RequestException as req_err:
        logger.error(
            f"An error occurred: {req_err}"
        )  # Catch-all for any other request-related errors
    except ValueError as json_err:
        logger.error(
            f"JSON decoding failed: {json_err}"
        )  # Issues with decoding the JSON response

    return response


def load_image(
    path: Path | str, image_data: Optional[Path | str | BytesIO] = None
) -> Image.Image:
    """

    Get image from path / Download image-data from URL and load into return variable.
    Takes pathlib.Path or url-string. Returns Pillow image object.

    """

    if not isinstance(path, Path) and is_hyperlink(path):
        response = get_api_data(path)
        image_data = BytesIO(response.content)
    else:
        image_data = path

    img = Image.open(image_data)

    return img


def load_dataframe(
    file_path: Path | str,
    datafiles: DataFiles,
    delimiter: str = ",",
    request_headers: dict = {},
    json_nests: list[str] = [],
) -> pd.DataFrame:
    """

    Loads a file into a pandas DataFrame. Supports CSV, TXT, and Parquet files.

    Takes pathhlib Path, optional keyword argument for csv delimiter
    returns dataframe
    raises ValueError if file doesn't exist.

    """

    if isinstance(file_path, str):
        if file_path.endswith(".txt"):
            return pd.read_csv(file_path)
        else:
            file = get_api_data(file_path, headers=request_headers)
            parsed_file = datafiles.parse_json(file.json(), json_nests)
            normalized_df = pd.json_normalize(parsed_file)
            return normalized_df

    if not file_path.is_file():
        raise FileNotFoundError(
            f"File {file_path.name} not found in {file_path.parents[0]}. Make sure the filename is correctly defined in config.toml"
        )

    file_extension = file_path.suffix

    if file_extension == ".csv":
        return pd.read_csv(file_path, delimiter=delimiter)
    elif file_extension == ".txt":
        return pd.read_csv(file_path)
    elif file_extension == ".parq" or file_extension == ".parquet":
        return pd.read_parquet(file_path)
    elif file_extension == ".json":
        with open(file_path) as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported file type: {file_extension}. Supported types: csv, txt, parq/parquet, json."
        )


with open((Path(__file__).parent / "../../config.toml").resolve(), "rb") as f:
    config = tomllib.load(f)
