from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
import matplotlib.pyplot as plt
from requests import get
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import inspect
import json
import re
import logging
import json
import statistics
from scipy.stats import norm
import random
import colorsys

# @dataclass
class MessageFileLoader():
    file_stem: str
    message_file_paths: list[Path]
    datafiles: list[pd.DataFrame]
    images: list[Image]
    
    def __init__(self, **kwargs):
        # Returns location of file initiating this class
        caller = inspect.stack()[1][1]

        # Getting file name
        self.file_stem = Path(caller).stem

        if not self.file_stem in config:
            raise ValueError(f"No variables set for file {self.file_stem}")
        
        # Add any settings from config.toml as attributes
        for key in config[self.file_stem]:
            setattr(self, key, config[self.file_stem][key])
        # Add any keyword arguments passed to init as attributes
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        if not hasattr(self, "input"):
            raise ValueError(f"No inputfiles set for script {self.file_stem}")
        
        self.input_path = (Path("../..") / Path(config["settings"]["processed"])).resolve()
        self.message_file_paths = [(self.input_path / Path(message_file)).resolve() if not is_hyperlink(message_file) else message_file for message_file in self.input]
    
    def get_datafiles(self) -> list[pd.DataFrame]:
        df_list = [load_dataframe(file, loader=self) for file in self.message_file_paths]
        self.datafiles = df_list

        return df_list
    
    def parse_json(self, json_file: json) -> json:
        """
        
        Gets nested level belonging to the JSON file from config.toml.
        Uses self.file_stem as suffix.
        Takes and returns JSON. If no config is present, returns same JSON.
        
        """

        if hasattr(self, "nests"):
            for level in self.nests:
                json_file = json_file[level]
        
        return json_file
    
    def get_images(self):
        if hasattr(self, "images"):
            self.images = [load_image(image) for image in self.images]
        else:
            raise ValueError(f"images in table {self.file_stem} not found in config.toml")
        
        return self.images
    
    def clean_transform_data(self):
        pass

class PlotSettings(BaseModel):
    """Base settings for plotting."""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: tuple = (10, 6)
    rotation: int = 0
    legend_title: Optional[str] = None

class BasePlot:
    """Base class for creating plots."""
    def __init__(self, settings: PlotSettings):
        self.settings = settings
        self.fig = None
        self.ax = None
        
    def create_figure(self, loader: Optional[MessageFileLoader]):
        """Create a figure and configure it based on settings."""
        # IMPROVEMENT: Instead of hardcoding figsize=(10, 6) everywhere,
        # we use the value from settings
        self.fig, self.ax = plt.subplots(figsize=self.settings.figsize or loader.figsize)
        
        # IMPROVEMENT: Instead of repeating these calls in every plotting function,
        # we centralize them here once
        try:
            self.ax.set_xlabel(self.settings.xlabel or loader.xlabel)
            self.ax.set_ylabel(self.settings.ylabel or loader.ylabel)
            self.ax.set_title(self.settings.title or loader.title)
        except:
            raise ValueError("Settings class or config.toml have to be set to plot xlabel, ylabel and title.")
        
        # IMPROVEMENT: Legend settings are now configurable
        if self.settings.legend_title is not None:
            self.ax.legend(title=self.settings.legend_title)
            
        plt.tight_layout()
        return self.fig, self.ax
    
        # this helps us use the figure in other classes
        # this is the Open-Closed principle;
        # make code CLOSED for modification (is essence, we will probably never 
        # need to modify the BasePlot class) but OPEN for extension if we want to
        # add more features
    def get_figure(self):
        """Return the figure, creating it if needed."""
        if self.fig is None:
            self.create_figure()
        return self.fig

def is_hyperlink(path: str) -> bool:
    return bool(re.search(r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$", path))

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

def get_birthdays(birthday: pd.Timestamp, from_date: pd.Timestamp, to_date: pd.Timestamp):

    date_range = pd.date_range(start=from_date, end=to_date)

    day = birthday.day
    month = birthday.month

    birthdays = date_range[(date_range.day == day) & (date_range.month == month)].floor("d")

    return birthdays

def create_birthday_list(author, birthday, message_data):

    first_message = message_data.loc[message_data["author"] == author]["timestamp"].min()
    last_message = message_data["timestamp"].max()

    birthdays = pd.DataFrame(index=get_birthdays(birthday, first_message, last_message))

    birthdays["author"] = author

    return birthdays

def how_many_congratulations(message_data, author, birthdays, congratulations_regex):

    first_message = message_data.loc[message_data["author"] == author]["timestamp"].min()
    last_message = message_data["timestamp"].max()

    congratulations = message_data.loc[(message_data["message"].str.match(congratulations_regex)) & \
                                        (message_data["author"] == author)]["timestamp"].dt.floor("D").to_list()

    if len(birthdays) < 1:
        return 0, 0

    birthdays = birthdays.reset_index(names="timestamp")

    birthdays = birthdays.loc[(birthdays["author"] != author) & (birthdays["timestamp"] >= first_message) & (birthdays["timestamp"] <= last_message)]

    congratulated = len(birthdays.loc[birthdays["timestamp"].isin(congratulations)])

    not_congratulated = len(birthdays.loc[~birthdays["timestamp"].isin(congratulations)])

    return congratulated, not_congratulated

def birthday_congratulations(message_data: pd.DataFrame, birthday_json: json, congratulations_regex: re.match):

    authors = message_data["author"].unique()

    birthday_df = pd.DataFrame()

    for author in authors:

        if author not in birthday_json:
            logging.warning(f"No birthday for {author} in birthday-JSON, continuing...")
            continue
        
        birthday_list = create_birthday_list(author, birthday_json[author], message_data)

        if birthday_df.empty:
            birthday_df = birthday_list
        else:
            birthday_df = pd.concat([birthday_df, birthday_list])
    
    birthday_list.sort_index(inplace=True)
    
    congratulated_df = pd.DataFrame(columns=["author", "congratulated", "not_congratulated"])
    
    for author in authors:
        congratulated, not_congratulated = how_many_congratulations(message_data, author, birthday_df, congratulations_regex)

        congratulated_df.loc[len(congratulated_df)] = [author, congratulated, not_congratulated]

    return congratulated_df

def create_regex(regex, flags=re.I|re.U):
    return re.compile(regex, flags)

def calculate_age(birthday: pd.Timestamp):

    return (datetime.now() - birthday) / pd.Timedelta(365.25, "d")

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
        if loader is not None and hasattr(loader, "request_headers"):
            headers = loader.request_headers
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
    elif file_extension == ".json":
        with open(file_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: csv, txt, parq/parquet.")

def create_distribution(datapoints: np.array):
    mean = statistics.mean(datapoints) 
    sd = statistics.stdev(datapoints)

    pdf = norm.pdf(datapoints, mean, sd)

    bell_values_x = np.linspace(mean - 3*sd, mean + 3*sd, 1000)
    bell_values_y = norm.pdf(bell_values_x, mean, sd)

    return [[datapoints, pdf], [bell_values_x, bell_values_y]]

def get_random_color():
    """ Returns random high saturation color """
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]

    return (r, g, b)


with open(Path("../../config.toml").resolve(), "rb") as f:
    config = tomllib.load(f)