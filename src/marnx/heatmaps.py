import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from settings import MessageFileLoader, BasePlot

class HeatmapLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        super().get_datafiles()
        self.clean_transform_data()

    def clean_transform_data(self):
        self.datafile[0]['hour'] = self.datafile['timestamp'].dt.hour
        self.datafile[0]['day_of_week'] = self.datafile['timestamp'].dt.dayofweek

        heatmap_data = self.datafile[0].pivot_table(index='day_of_week', columns='hour', aggfunc='size', fill_value=0)

        self.datafiles[0] = heatmap_data

class HeatmapPlotter(BasePlot):
    def __init__(self):
        super().__init__()
    
    def plot(self, loader):
        sns.heatmap(self.datafiles[0], cmap=loader.cmap, linewidths=loader.linewidths, annot=loader.annot, fmt=loader.fmt, cbar_kws=loader.cbar_kws)

        # Set y-axis tick labels
        plt.yticks(ticks=loader.yticks_ticks, labels=loader.yticks_labels, rotation=self.settings.rotation)

        plt.show()

def main():

    loader = HeatmapLoader()

    plotter = HeatmapPlotter()

    plotter.plot(loader)

if __name__ == "__main__":
    main()