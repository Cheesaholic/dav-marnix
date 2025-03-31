import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ppca import PPCA

# Local imports
from settings import (
    MessageFileLoader, 
    PlotSettings, 
    BasePlot, 
    df_contains_multiple_regexes,
    remove_url,
    remove_emoji,
    remove_image,
    roberta_spellcheck_on_series,
    get_random_color
)


class SpellingLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        # super().get_datafiles()
        self.clean_transform_data()

    def clean_transform_data(self):

        for key, value in self.datafiles.all():
            value["file"] = key

        self.datafiles.chat = pd.concat(self.datafiles.all(values=True), ignore_index=True)

        spelling_regexes = [self.spellcheck[x]["regex"] for x in self.spellcheck]

        self.datafiles.chat = df_contains_multiple_regexes(self.datafiles.chat, spelling_regexes, message_col=self.message_col)

        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col].str.split(self.sentence_regex)
        self.datafiles.chat = self.datafiles.chat.explode(self.message_col + "_split")

        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col + "_split"].apply(remove_url)
        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col + "_split"].apply(remove_image)
        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col + "_split"].apply(remove_emoji)
        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col + "_split"].str.strip()
        self.datafiles.chat = self.datafiles.chat.loc[self.datafiles.chat[self.message_col + "_split"].str.count(' ') >= self.sentence_min_token_len -1]
        self.datafiles.chat.drop_duplicates(inplace=True, ignore_index=True)

        roberta_output = roberta_spellcheck_on_series(self.datafiles.chat, self.roberta_model, self.spellcheck, author_col=self.author_col, message_col=self.message_col + "_split", file_col="file")

        roberta_output.to_parquet("roberta_outpt.parq")

        roberta_grouped = roberta_output.groupby([self.author_col, "file", "test"]).agg({
            "correct" : ["count", "sum"]
        })["correct"]

        roberta_grouped["pct"] = roberta_grouped["sum"] / roberta_grouped["count"]

        roberta_grouped = roberta_grouped.unstack(level=1)
        
        x = roberta_grouped["pct"].values
        x = StandardScaler().fit_transform(x)

        ppca = PPCA()

        ppca.fit(data=x, d=2, verbose=True)

        principal_df = pd.DataFrame(data = ppca.transform()
             , columns = ['PPC1', 'PPC2'])
        
        roberta_grouped.to_parquet("group.parq")
        principal_df.to_parquet("principal.parq")

        grouped = pd.read_parquet("group.parq")
        processed = pd.read_parquet("principal.parq")

        self.datafiles.grouped = grouped
        self.datafiles.processed = processed



class SpellingPlotter(BasePlot):
    def __init__(self, settings):
        super().__init__(settings)
    
    def plot(self, loader):
        super().create_figure(loader=loader)

        for target in loader.datafiles.grouped.index.get_level_values(1).str.replace(r"_\d", "").unique():
            indicesToKeep = loader.datafiles.grouped.index.get_level_values(1).str.replace(r"_\d", "") == target
            self.ax.scatter(loader.datafiles.processed.loc[indicesToKeep, 'PPC1']
                    , loader.datafiles.processed.loc[indicesToKeep, 'PPC2'], c = get_random_color(hex=True), s = 50)

        self.fig.suptitle(loader.suptitle, fontsize=18, y=0.99)

        self.ax.set_title(loader.title.format(tests = ", ".join([x for x in loader.spellcheck])), fontsize=10, y=0.995)

        plt.show()

def main():

    loader = SpellingLoader()

    plotter = SpellingPlotter(PlotSettings())

    plotter.plot(loader)

if __name__ == "__main__":
    main()