import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, remove_emoji, remove_image,
                             remove_numbers, remove_url)


class PeriodSettings(PlotSettings):
    min_tokens: int
    min_long_messages: int
    end_of_sentence_list: list[str]
    category_label_y_offset: float
    category_label_weight: str


class PeriodLoader(MessageFileLoader):
    settings: PeriodSettings

    def __init__(self, settings: PeriodSettings):
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):

        self.datafiles.chat = self.datafiles.merge(capitalize_filename=True)

        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)
            .apply(remove_image)
            .apply(remove_emoji)
            .apply(remove_numbers)
            .str.strip()
        )

        self.datafiles.chat["endswith_period"] = self.datafiles.chat[
            self.settings.message_col
        ].str.endswith(tuple(self.settings.end_of_sentence_list))

        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat[self.settings.message_col].str.count(" ") + 1
            > self.settings.min_tokens
        ]

        df = self.datafiles.chat.groupby(["file", "author", "endswith_period"]).count()[
            ["timestamp"]
        ]

        df["cnt_long_messages"] = df.groupby(["file", "author"])["timestamp"].transform(
            "sum"
        )

        # Bereken het percentage per rij
        df["pct_endswith_period"] = (df["timestamp"] / df["cnt_long_messages"]) * 100

        # Filter alleen de rijen waarbij endswith_period == True
        result = df[df.index.get_level_values("endswith_period")]

        # Houd alleen de gewenste kolommen over
        result = result.drop(columns=["timestamp"])
        result.reset_index(level=["endswith_period", "author"], drop=True, inplace=True)
        result.reset_index(inplace=True)

        self.datafiles.processed = result


class PeriodPlotter(BasePlot):
    settings: PeriodSettings

    def __init__(self, settings: PeriodSettings):
        super().__init__(settings)

    def plot(self, data, **kwargs):
        super().create_figure(
            end_of_sentence_list=" ".join(self.settings.end_of_sentence_list), **kwargs
        )

        data = data.loc[data["cnt_long_messages"] >= self.settings.min_long_messages]

        sns.stripplot(
            data=data,
            x="pct_endswith_period",
            y="file",
            hue="file",
            jitter=True,
            ax=self.ax,
        )

        labels = [x.get_text() for x in self.ax.get_yticklabels()]

        self.ax.get_yaxis().set_visible(False)

        for collection_num in range(len(self.ax.collections)):

            # Get all the plotted points (as matplotlib PathCollections)
            points = self.ax.collections[collection_num].get_offsets()

            color = self.ax.collections[collection_num].get_facecolor()[0]

            # Find the point with the lowest x-value
            highest_point = min(points, key=lambda p: p[0])
            x_coord, y_coord = highest_point

            # Optional: label it
            self.ax.text(
                x_coord,
                y_coord + self.settings.category_label_y_offset,
                labels[collection_num],
                ha="center",
                color=color,
            ).set_fontweight(self.settings.category_label_weight)

        plt.show()


def main():

    settings = PeriodSettings(**AllVars())

    loader = PeriodLoader(settings)

    plotter = PeriodPlotter(settings)

    plotter.plot(loader.datafiles.processed)


if __name__ == "__main__":
    main()
