# File still needs updating to new framework...







import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@dataclass
class 

import tomllib
configfile = Path("config.toml").resolve()
with configfile.open("rb") as f:
    config = tomllib.load(f)
config

root = Path("").resolve()
processed = root / Path(config["processed"])
datafile = processed / config["current"]
if not datafile.exists():
    logger.warning(f"{datafile} does not exist. First run src/preprocess.py, and check the timestamp!")

df = pd.read_parquet(datafile)

# Extract hour and day of the week
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek


# --- 1. Create a Circular Heatmap ---
heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
radii = np.arange(7)

for i, (day, row) in enumerate(heatmap_data.iterrows()):
    ax.bar(theta, row, width=0.25, bottom=i, color=plt.cm.viridis(row / row.max()))

ax.set_xticks(theta)
ax.set_xticklabels([f"{h}:00" for h in range(24)])
ax.set_yticks(radii)
ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_title("Circular Heatmap of Message Activity", fontsize=14)
plt.show()


# --- 2. Create a Weekly activity heatmap ---

# Create a pivot table counting messages per hour and day of the week
heatmap_data = df.pivot_table(index='day_of_week', columns='hour', aggfunc='size', fill_value=0)

# Set up the matplotlib figure
plt.figure(figsize=(12, 6))

# Create the heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5, annot=True, fmt='d', cbar_kws={'label': 'Number of Messages'})

# Set axis labels and title
plt.title('Weekly Message Activity Heatmap')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')

# Set y-axis tick labels
plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=0)

# Show the plot
plt.show()
