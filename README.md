# Marn_X - Marnix Ober 1890946

This project is set up to perform the exercises contained in the module Data Analysis & Visualization of the Master of Informatics - Applied Data Science ([MADS-DAV](https://github.com/raoulg/MADS-DAV)). The project tries to be as modular as possible, with almost all variables and settings being alterable without editing the code. Each Python module reads its settings from a corresponding table in the `config.toml` file in the main directory. Data can also be passed to the preprocessors and plotters by passing them to these classes.

## Features

- **

- **Modular Configuration:** Settings are organized in the `config.toml` file, with each table corresponding to a specific Python module. This structure promotes clarity and ease of maintenance.

- **Dynamic Attribute Loading:** The `MessageFileLoader` class dynamically loads configuration variables as class attributes, facilitating intuitive access within the codebase.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Cheesaholic/dav-marnix.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd dav-marnix
   ```

3. **Install Dependencies:**

   ```bash
   uv sync
   ```

## Quick Run

   To run all visualizations available by default, run:

   ```bash
   uv run marn_x
   ```

## Project Organization
```
dav-marnix/
├── config.toml                     # Configuration file for project parameters
├── lefthook.yml                    # Configuration for git hooks
├── pyproject.toml                  # Project metadata and dependency management
├── uv.lock                         # Lock file for dependency installs
├── README.md                       # Project overview and usage instructions

├── img/                            # Official project images to be handed in

├── models/                         # Bertopic models, generated with TopicLoader

├── data/                           # in- and output for loaders and plotters
│   ├── raw/                        # Data files to be loaded
│   └── processed/                  # Output plots

├── src/                            # Source code modules
│   ├── <project files>.py          # Files per lesson
│   ├── main.py                     # Function to run all project files
│   └── utils/                      # Utility submodules
│       ├── data_transformers.py    # Functions for transforming data
│       ├── file_operations.py      # Functions for loading from files
│       └── plot_styling.py         # Functions for styling plots

```

## Classes Available

### AllVars
The AllVars class loads all global settings from the `config.toml` file in the main directory of the project. These global settings are then overwritten by specific settings from individual files (if defined). These specific settings can also be overwritten by passing settings with the same name to the class when initializing. See [Customize your Experience](#customize-your-experience) below for examples.

### GeneralSettings(BaseModel)
Settings that almost every file needs.

### PlotSettings(GeneralSettings)
Settings specific for plotting. These contain a few standard values that can be overwritten by using the AllVars class.

### IndividualSettings(PlotSettings)
Not acually called IndividualSettings, but has another name specific to the file. These settings are almost always only used by one plot.

### MessageFileLoader
Parses dict with datafiles passed as `input` setting and loads datafiles into datafiles attribute (as DataFiles class).

### DataFiles
Initialized by `MessageFileLoader`, loads datafiles (ex. csv, parq, txt, json) and makes them available as attributes to this class as Pandas DataFrames (except for json-objects which are dicts).

### IndividualLoader(MessageFileLoader)
Not acually called IndividualLoader, but has another name specific to the file. Takes `IndividualSettings` object when initializing. Performs preprocessing specific for file. Almost always outputs to attribute 'processed'.

### BasePlot
Creates general environment that almost all plots need.
Method `create_figure` sets almost all characteristics of the plot. This method uses (child of) `PlotSettings` object to edit Matplotlib Figure and Axes settings.

### IndividualPlot(BasePlot)
Not acually called IndividualPlot, but has another name specific to the file. Takes `IndividualSettings` object when initializing. Calls `BasePlot` to create Matplotlib environment. Performs actions specific to the file. Can be passed settings to change them last-minute.

## Customize your Experience

1. **Configure Settings:**

   Define your settings in the `config.toml` file. Each table should correspond to a Python file name. All variables passed here are available to the corresponding preprocessors and plotters. Example:

   ```toml
   [ file_name ]
   setting1 = "value1"
   setting2 = "value2"
   setting3 = 1970-01-01
   setting4 = ["value3", "value4"]
   # settings.setting3 -> date(1970, 1, 1)
   ```

2. **Load DataFiles:**

   In the same `config.toml` file, in front of or after the settings, pass 'data files' to the 'input' setting. Place files with filenames  corresponding to the values you enter into the 'raw' folder. The location of the 'raw' folder is defined in the `config.toml`. The `AllVars` class will try to load your datafile and make it available as an attribute to individual children of the MessageFileLoader class.

   ```toml
   [ file_name ]
   setting1 = "value1", # etc....
   
   input.chat = "friends.csv"
   # <loader>.datafiles.chat available as Pandas DataFrame
   ```

3. **Load Variables with AllVars:**

   Use the the `AllVars` class to load variables.
   When initializing, the class will try to load variables from the `config.toml` corresponding to the filename calling the class. If none are available: please pass all necessary attributes to the `AllVars` class to ensure the pipeline runs smoothly.

   #### myfile.py
   ```python
   from marn_x.settings import AllVars

   variables = AllVars(setting1="foo")
   # Loading all global settings from config.toml, overwriting with specifics from section [ myfile ], overwriting with variables passed to AllVars
   ```

   When loading from a Jupyter Notebook, or to load specifics from a file with another filename than the caller, pass the filename corresponding with the required settings to the `file_stem` setting, when initializing AllVars:

   #### other_file_without_settings.py

   ```python
   from marn_x.settings import AllVars

   variables = AllVars(setting1="foo", file_stem="myfile")
   # Loading all global settings from config.toml, overwriting with specifics from section [ myfile ], overwriting with variables passed to AllVars
   ```
3. **Load Settings:**

   AllVars can be passed as a dict (using **) to (selfmade) settings classes that inherit PlotSettings. Example:

   #### myfile.py
   ```python
   class MySettings(PlotSettings):
      setting1: str
   
   settings = MySettings(**AllVars(setting1="foo"))
   # Loading all global settings from config.toml, overwriting with specifics from section [ myfile ], overwriting with variables passed to AllVars
   # Loading all variables into class MySettings
   ```

4. **Load Data with MessageFileLoader**

   When setting up a config.toml file as described in step 2, the file is available in the following way:
   #### myfile.py
   ```python
   class MySettings(PlotSettings):
      setting1: str # etc..

   class MyLoader(MessageFileLoader):
      settings: MySettings

      def __init__(self, settings: MySettings):
         super().__init__(settings)

   
   def main():
      settings = MySettings(AllVars())
      loader = MyLoader(settings)

      print(loader.datafiles.chat) 
   # pd.DataFrame containing data from friends.csv
   ```


5. **Generate Visualizations:**

   Implement your visualization logic using the loaded configurations. For example:

   #### config.toml
   ```toml
   [ file_name ]
   title = "My Title"
   xlabel = "xvalue"
   ylabel = "yvalue"
   ```
   #### myfile.py
   ```python
   ...

   class IndividualPlotter(BasePlot):
      settings: IndividualSettings

      def __init__(self, settings: IndividualSettings):
         super().__init__(settings)

      def plot(self, data, **kwargs):
         super().get_figure(**kwargs)

         self.ax.plot(data.x, data.y)
         
         plt.show()
         self.to_png()

   # Plot with title: My Title, x_label: xvalue, y_label: yvalue and data from data gets shown to user and saved to png
   ```

## Contributing

We welcome contributions to enhance this project. To contribute:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature-branch
   ```

3. Commit your changes:

   ```bash
   git commit -m 'Add new feature'
   ```

4. Push to the branch:

   ```bash
   git push origin feature-branch
   ```

5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Python TOML](https://realpython.com/python-toml/) for insights on using TOML in Python.

- [Best README Template](https://github.com/othneildrew/Best-README-Template) for inspiration on structuring this README.

For more information on creating effective README files, consider visiting [Make a README](https://www.makeareadme.com/).
