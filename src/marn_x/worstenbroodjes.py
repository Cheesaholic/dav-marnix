import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Local import
from settings import MessageFileLoader, BasePlot, get_carnaval


class WorstenbroodjesLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        super().get_datafiles()
        super().get_images()
        self.clean_transform_data()

    def clean_transform_data(self):

        df = self.datafiles.wikipedia_api

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], format=self.api_dt_format)

        agg = df.groupby([pd.Grouper(key=self.timestamp_col, freq='W-MON')])['views'] \
                .sum() \
                .reset_index() \
                .sort_values(self.timestamp_col) \
                .set_index(self.timestamp_col) \

        self.datafiles.wikipedia_api = agg

class WorstenbroodjesPlotter(BasePlot):
    def __init__(self):
        super().__init__()
    
    def plot(self, loader):
        loader.datafiles.wikipedia_api.plot.line(linewidth=5, figsize=(20,10), color="#5C4033", legend=False, ax=self.ax)

        # Add Carnaval dates as avxspan
        for instance in get_carnaval():
            self.ax.axvspan(instance["start"], instance["end"], facecolor='black', alpha=0.3)
        
        # Add annotations
        self.ax.annotate("Pagina-bezoeken per week", (loader.datafiles.wikipedia_api.index.max(), loader.datafiles.wikipedia_api.iloc[-1][0]), fontsize=12)
        self.ax.annotate("Peter Gillis veroordeeld voor belastingfraude -->", (datetime(2023, 6, 6), loader.datafiles.wikipedia_api["views"].max()), fontsize=13)

        # Add background image
        self.ax.imshow(loader.get_images()[0], extent=[loader.datafiles.wikipedia_api.index.min(), loader.datafiles.wikipedia_api.index.max(), 0, loader.datafiles.wikipedia_api["views"].max()], alpha=0.2, aspect='auto')

        # Hide x-axis label to reduce clutter
        self.ax.get_xaxis().get_label().set_visible(False)

        # Set title
        self.ax.set_title("Wikipedia-pagina 'Worstenbroodje' populair rond ", fontdict={'fontsize': 40, 'fontweight': 'medium'}, loc="left", pad=20)

        # Get title characteristics to add Bbox to last word
        title = self.ax.title
        title_pos = title.get_position()
        title_bbox = title.get_window_extent()

        # Get coordinates for bbox on last word
        last_word_x = title_bbox.x1 / self.fig.dpi * 0.05 + 0.3  # Unknown offset needed for coordinates. Will check why in later stage
        last_word_y = title_pos[1] - 0.077 # Unknown offset needed for coordinates. Will check why in later stage

        # Add the last word with bbox
        self.ax.text(last_word_x, last_word_y, "Carnaval", fontsize=40, fontweight='medium',
                bbox=dict(facecolor='black', edgecolor='none', alpha=0.3),
                transform=self.fig.transFigure, ha='left', va='center')
        
        # Hide axis lines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        plt.show()

def main():

    loader = WorstenbroodjesLoader()

    plotter = WorstenbroodjesPlotter()

    plotter.plot(loader)

if __name__ == "__main__":
    main()