import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Local imports
from settings import MessageFileLoader, PlotSettings, BasePlot, birthday_congratulations, calculate_age, create_regex


class DistributionLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        # super().get_datafiles()
        self.clean_transform_data()

    def clean_transform_data(self):

        self.datafiles.chat = self.datafiles.chat.loc[self.datafiles.chat[self.author_col].map(self.datafiles.chat.groupby(self.author_col)[self.message_col].count() >= self.min_messages) == True]

        birthday_dict = {k : pd.to_datetime(v, format=self.birthday_dateformat) for k, v in self.datafiles.birthdates.items()}

        congratulations_df = birthday_congratulations(self.datafiles.chat, birthday_dict, create_regex(self.congratulations_regex))
        
        congratulations_df[self.xlabel] = [calculate_age(birthday_dict[x]) if x in birthday_dict else 0 for x in congratulations_df[self.author_col]]

        congratulations_df[self.ylabel] = (congratulations_df["congratulated"] / (congratulations_df["congratulated"] + congratulations_df["not_congratulated"])) * 100

        congratulations_df[self.huelabel] = [self.datafiles.genders[x] if x in self.datafiles.genders else "?" for x in congratulations_df[self.author_col]]

        congratulations_df = congratulations_df.loc[congratulations_df[self.xlabel] > 0]

        self.datafiles.processed = congratulations_df


class DistributionPlotter(BasePlot):
    def __init__(self, settings):
        super().__init__(settings)
    
    def plot(self, loader):
        super().create_figure(loader=loader)

        sns.scatterplot(x=loader.xlabel, y=loader.ylabel, data=loader.datafiles.processed, hue=loader.huelabel, ec=None, ax=self.ax)

        b, a = np.polyfit(loader.datafiles.processed[loader.xlabel], loader.datafiles.processed[loader.ylabel], deg=1)
        xseq = np.linspace(loader.datafiles.processed[loader.xlabel].min(), loader.datafiles.processed[loader.xlabel].max())

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