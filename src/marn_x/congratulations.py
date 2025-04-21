import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Local imports
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, author_min_messages,
                             birthday_congratulations, calculate_age,
                             create_regex, get_linear_regression)


class FelicitationSettings(PlotSettings):
    marker_size: int | float
    hue_label: str
    hue_dict: dict
    min_messages: int
    birthday_dateformat: str
    congratulations_regex: str
    congratulations_flags: str


class FelicitationLoader(MessageFileLoader):
    settings: FelicitationSettings

    def __init__(self, settings: FelicitationSettings):
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):

        self.datafiles.chat = author_min_messages(
            self.datafiles.chat,
            self.settings.author_col,
            self.settings.message_col,
            self.settings.min_messages,
        )

        birthday_dict = {
            k: pd.to_datetime(v, format=self.settings.birthday_dateformat)
            for k, v in self.datafiles.birthdates.items()
        }

        congratulations_df = birthday_congratulations(
            self.datafiles.chat,
            birthday_dict,
            create_regex(self.settings.congratulations_regex),
        )

        congratulations_df[self.settings.xlabel] = [
            calculate_age(birthday_dict[x]) if x in birthday_dict else 0
            for x in congratulations_df[self.settings.author_col]
        ]

        congratulations_df[self.settings.ylabel] = (
            congratulations_df["congratulated"]
            / (
                congratulations_df["congratulated"]
                + congratulations_df["not_congratulated"]
            )
        ) * 100

        congratulations_df[self.settings.hue_label] = [
            self.datafiles.hue[x] if x in self.datafiles.hue else "?"
            for x in congratulations_df[self.settings.author_col]
        ]

        congratulations_df["hue"] = congratulations_df[self.settings.hue_label].map(
            self.settings.hue_dict
        )

        congratulations_df = congratulations_df.loc[
            congratulations_df[self.settings.xlabel] > 0
        ]

        self.datafiles.processed = congratulations_df


class FelicitationPlotter(BasePlot):
    settings: FelicitationSettings

    def __init__(self, settings: FelicitationSettings):
        super().__init__(settings)

    def plot(self, data, **kwargs):
        super().get_figure(**kwargs)

        print(data)

        sns.scatterplot(
            x=self.settings.xlabel,
            y=self.settings.ylabel,
            data=data,
            hue=self.settings.hue_label,
            palette=data["hue"].unique().tolist(),
            ec=None,
            ax=self.ax,
            s=self.settings.marker_size,
            legend=False if self.settings.hide_legend else "auto",
        )

        get_linear_regression(
            self.ax, data[self.settings.xlabel], data[self.settings.ylabel]
        )

        plt.show()
        self.to_png()


def main():

    settings = FelicitationSettings(**AllVars())

    loader = FelicitationLoader(settings)

    plotter = FelicitationPlotter(settings)

    outlier = loader.datafiles.processed[
        loader.datafiles.processed[settings.ylabel]
        == loader.datafiles.processed[settings.ylabel].max()
    ].iloc[0]

    plotter.plot(
        loader.datafiles.processed,
        annotation_x=outlier[settings.xlabel],
        annotation_y=outlier[settings.ylabel],
    )


if __name__ == "__main__":
    main()
