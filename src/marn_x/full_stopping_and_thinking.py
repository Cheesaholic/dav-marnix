from typing import Literal

import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, create_colored_annotation,
                             remove_emoji, remove_image, remove_numbers,
                             remove_url)
from marn_x.utils.plot_styling import stripplot_mean_line


class PeriodSettings(PlotSettings):
    markers: list[str]
    marker_size: int | float
    min_tokens: int
    min_long_messages: int
    end_of_sentence_list: list[str]
    annotation_location_highest_lowest: Literal["higest", "lowest"]
    annotation_location_x_y: Literal["x", "y"]
    mean_color: str
    mean_alpha: float | int


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
        super().get_figure(
            end_of_sentence_list=" ".join(self.settings.end_of_sentence_list), **kwargs
        )

        # Filter long messages (for future use in dashboard)
        data = data.loc[data["cnt_long_messages"] >= self.settings.min_long_messages]

        sns.stripplot(
            data=data,
            x="pct_endswith_period",
            y="file",
            hue="file",
            size=self.settings.marker_size,
            jitter=True,
            ax=self.ax,
        )

        stripplot_mean_line(
            self.ax,
            data,
            "pct_endswith_period",
            "file",
            color=self.settings.mean_color,
            alpha=self.settings.mean_alpha,
        )

        labels = [x.get_text() for x in self.ax.get_yticklabels()]

        for collection_num in range(len(self.ax.collections)):

            create_colored_annotation(
                self.ax,
                labels[collection_num],
                self.ax.collections[collection_num].get_offsets(),
                highest_lowest=self.settings.annotation_location_highest_lowest,
                x_y=self.settings.annotation_location_x_y,
                color=self.ax.collections[collection_num].get_facecolor()[0],
                x_offset=self.settings.annotation_x_offset,
                y_offset=self.settings.annotation_y_offset,
                fontweight=self.settings.annotation_fontweight,
            )

            # Set marker style
            marker_obj = mmarkers.MarkerStyle(self.settings.markers[collection_num])
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            self.ax.collections[collection_num].set_paths(
                [path] * len(self.ax.collections[collection_num].get_offsets())
            )

        plt.show()
        self.to_png()


def main():

    settings = PeriodSettings(**AllVars())

    loader = PeriodLoader(settings)

    plotter = PeriodPlotter(settings)

    plotter.plot(loader.datafiles.processed)


if __name__ == "__main__":
    main()
