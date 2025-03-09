from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import matplotlib.pyplot as plt
from requests import get
import tomllib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
import pandas as pd
from PIL import Image
from io import BytesIO
import inspect
import json
import re
import logging

# @dataclass
class MessageFileLoader():
    file_stem: str
    message_file_paths: list[Path]
    datafiles: list[pd.DataFrame]
    images: list[Image]
    
    def __init__(self):
        # Returns location of file initiating this class
        caller = inspect.stack()[1][1]

        # Getting file name
        self.file_stem = Path(caller).stem
        input_file = "input_" + self.file_stem
        
        if not "input_" + self.file_stem in config:
            raise ValueError(f"No inputfiles set for script {self.file_stem}")
        
        message_files = config["input_" + self.file_stem]
        self.input_path = (Path("../..") / Path(config["processed"])).resolve()
        self.message_file_paths = [(self.input_path / Path(message_file)).resolve() if not is_hyperlink(message_file) else message_file for message_file in message_files]
    
    def get_datafiles(self) -> list[pd.DataFrame]:
        df_list = [load_dataframe(file, loader=self) for file in self.message_file_paths]
        self.datafiles = df_list

        return df_list
    
    def get_request_headers(self) -> dict:
        if "request_headers_" + self.file_stem in config:
            return config["request_headers_" + self.file_stem]
        else:
            return {}
    
    def parse_json(self, json_file: json) -> json:
        """
        
        Gets nested level belonging to the JSON file from config.toml.
        Uses self.file_stem as suffix.
        Takes and returns JSON. If no config is present, returns same JSON.
        
        """

        nests = "json_nests_" + self.file_stem

        if nests in config:
            for level in config[nests]:
                json_file = json_file[level]
        
        return json_file
    
    def get_images(self):
        if "images_" + self.file_stem in config:
            images = config["images_" + self.file_stem]
            image_list = [load_image(image) for image in images]
        else:
            raise ValueError(f"images_{self.file_stem} not found in config.toml")
        
        self.images = image_list
        return image_list
    
    def clean_transform_data(self):
        pass

class BasePlot:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots(figsize=(20,10))
    
    def plot(self):
        pass

def is_hyperlink(path: str) -> bool:
    return bool(re.search(r"^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$", path))

def is_datafile_exists(path: Path) -> bool:
    """
    Checks if a entered path,file combination exists. 

    Returns bool.
    """

    if path.is_file():
        return True
    else:
        return False

def get_carnaval():
    return [
        {"year": 2023,
         "start": datetime(2023,2,19),
         "end": datetime(2023,2,21)},
        {"year": 2024,
         "start": datetime(2024,2,11),
         "end": datetime(2024,2,13)},
        {"year": 2025,
         "start": datetime(2025,3,1),
         "end": datetime(2025,3,4)}]

def get_api_data(endpoint: str, headers: dict = {}, payload: dict = {}) -> list:
    """ Connects to API via endpoint and variables in string.
        Returns JSON-object. """
    
    logging.debug(f"API Call. Endpoint: {endpoint}, headers: {headers}, payload: {payload}")

    try:
        # Fire response. Throw error if request takes longer than 20 seconds
        response = get(endpoint, headers=headers, data=payload, timeout=20)

        logging.debug(f"API HTML response-code {response.status_code}")

        # Raises HTTPError for bad responses (4xx or 5xx)
        response.raise_for_status() 

        return response
    
    except HTTPError as http_err:
        logging.error(f'HTTP error occurred: {http_err}')  # e.g., 404 Not Found, 500 Internal Server Error
    except ConnectionError as conn_err:
        logging.error(f'Connection error occurred: {conn_err}')  # Issues with network connectivity
    except Timeout as timeout_err:
        logging.error(f'Timeout error occurred: {timeout_err}')  # Request timed out
    except RequestException as req_err:
        logging.error(f'An error occurred: {req_err}')  # Catch-all for any other request-related errors
    except ValueError as json_err:
        logging.error(f'JSON decoding failed: {json_err}')  # Issues with decoding the JSON response


def load_image(path: Path | str):
    """ 
    
    Get image from path / Download image-data from URL and load into return variable.
    Takes pathlib.Path or url-string. Returns Pillow image object. 
    
    """

    if is_hyperlink(path):
        response = get_api_data(path)
        image_data = BytesIO(response.content)
    else:
        image_data = path

    img = Image.open(image_data)

    return img


def load_dataframe(file_path: Path | str, loader: Optional[MessageFileLoader] = None, delimiter: str = ",") -> pd.DataFrame:
    """

    Loads a file into a pandas DataFrame. Supports CSV, TXT, and Parquet files.
    
    Takes pathhlib Path, optional keyword argument for csv delimiter
    returns dataframe
    raises ValueError if file doesn't exist.

    """
    
    if isinstance(file_path, str) and file_path.startswith("http"):
        headers = {}
        if loader is not None:
            headers = loader.get_request_headers()
        file = get_api_data(file_path, headers=headers)
        parsed_file = loader.parse_json(file.json())
        normalized_df = pd.json_normalize(parsed_file)
        return normalized_df

    if not is_datafile_exists(file_path):
        raise FileNotFoundError(f"File {file_path.name} not found in {file_path.parents[0]}. Make sure the filename is correctly defined in config.toml")
    
    file_extension = file_path.suffix
    
    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension == ".txt":
        return pd.read_csv(file_path, delimiter=delimiter)
    elif file_extension == ".parq" or file_extension == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: csv, txt, parq/parquet.")


with open(Path("../../config.toml").resolve(), "rb") as f:
    config = tomllib.load(f)