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
