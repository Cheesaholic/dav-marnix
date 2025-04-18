# Marnix Ober 1890946

This project provides a framework for creating visualizations based on configurations specified in `config.toml` files. Each Python module reads its settings from a corresponding table in the `config.toml`, ensuring a modular and organized approach to configuration management.

## Features

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

## Usage

1. **Configure Settings:**

   Define your settings in the `config.toml` file. Each table should correspond to a Python file name. For example:

   ```toml
   [module_name]
   setting1 = "value1"
   setting2 = "value2"
   ```

2. **Load Configurations:**

   Utilize the `MessageFileLoader` class to load configurations:

   ```python
   from settings import MessageFileLoader

   loader = MessageFileLoader()
   ```

3. **Generate Visualizations:**

   Implement your visualization logic using the loaded configurations. For example:

   ```python
   import matplotlib.pyplot as plt

   # Example data
   data = [1, 2, 3, 4, 5]

   # Plotting
   plt.plot(data)
   plt.title(config.plot_title)
   plt.show()
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
