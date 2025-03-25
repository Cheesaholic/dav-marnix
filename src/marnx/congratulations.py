import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Local imports
from settings import MessageFileLoader, PlotSettings, BasePlot, birthday_congratulations, calculate_age, create_regex


class DistributionLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        super().get_datafiles()
        self.clean_transform_data()

    def clean_transform_data(self):


        self.datafiles[0] = self.datafiles[0].loc[self.datafiles[0]["author"].map(self.datafiles[0].groupby("author")["message"].count() >= self.min_messages) == True]

        birthday_dict = {k : pd.to_datetime(v, format=self.birthday_dateformat) for k, v in self.datafiles[1].items()}

        congratulations_df = birthday_congratulations(self.datafiles[0], birthday_dict, create_regex(self.congratulations_regex))
        
        congratulations_df[self.xlabel] = [calculate_age(birthday_dict[x]) if x in birthday_dict else 0 for x in congratulations_df["author"]]

        congratulations_df[self.ylabel] = (congratulations_df["congratulated"] / (congratulations_df["congratulated"] + congratulations_df["not_congratulated"])) * 100

        congratulations_df[self.huelabel] = [self.datafiles[2][x] if x in self.datafiles[2] else "?" for x in congratulations_df["author"]]

        congratulations_df = congratulations_df.loc[congratulations_df[self.xlabel] > 0]

        self.datafiles.append(congratulations_df)


class DistributionPlotter(BasePlot):
    def __init__(self, settings):
        super().__init__(settings)
    
    def plot(self, loader):
        super().create_figure(loader=loader)

        sns.scatterplot(x=loader.xlabel, y=loader.ylabel, data=loader.datafiles[-1], hue=loader.huelabel, ec=None, ax=self.ax)

        b, a = np.polyfit(loader.datafiles[-1][loader.xlabel], loader.datafiles[-1][loader.ylabel], deg=1)
        xseq = np.linspace(loader.datafiles[-1][loader.xlabel].min(), loader.datafiles[-1][loader.xlabel].max())

        self.ax.plot(xseq, a + b * xseq, color="k", lw=1, alpha=0.5, label="LSRL")
        self.ax.legend()

        self.fig.suptitle(loader.suptitle, fontsize=18, y=0.99)

        self.ax.set_title(loader.title.format(congratulations_regex = loader.congratulations_regex), fontsize=10, y=0.995)

        plt.show()

def main():

    loader = DistributionLoader()

    plotter = DistributionPlotter(PlotSettings())

    plotter.plot(loader)

if __name__ == "__main__":
    main()