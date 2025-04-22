import json
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger
from PIL import Image
from requests import Response, get
from requests.exceptions import (ConnectionError, HTTPError, RequestException,
                                 Timeout)


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
        self.input_path = (Path(__file__).parent / "../../.." / raw).resolve()

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


def is_hyperlink(path: str | Path) -> bool:
    if isinstance(path, Path):
        return False
    return bool(
        re.search(
            r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$",
            str(path),
        )
    )


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
