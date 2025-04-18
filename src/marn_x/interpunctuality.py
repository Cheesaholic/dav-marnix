from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# Local imports
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, create_colored_annotation,
                             remove_edited, remove_emoji, remove_image,
                             remove_more_information, remove_removed,
                             remove_security_code, remove_url)


class DistributionSettings(PlotSettings):
    input_settings: dict
    norm_alpha: float | int
    hist_alpha: float | int
    interpunction_regex: str
    word_regex: str
    min_words: int
    norm_colnam: str = "ratio_interpunction"
    p_value_mw: Optional[float] = None
    p_value_ks: Optional[float] = None


class DistributionLoader(MessageFileLoader):
    settings: DistributionSettings

    def __init__(self, settings: DistributionSettings):
        super().__init__(settings)
        self.clean_transform_data()

    def clean_transform_data(self):

        if len(self.datafiles) < 2:
            raise ValueError(
                f"Only found 1 dataset in {self.settings.file_stem}. You need 2 for comparison"
            )

        self.datafiles.chat = self.datafiles.merge()

        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)
            .apply(remove_image)
            .apply(remove_emoji)
            .apply(remove_more_information)
            .apply(remove_security_code)
            .apply(remove_removed)
            .apply(remove_edited)
            .str.strip()
        )

        self.datafiles.chat["n_words"] = self.datafiles.chat[
            self.settings.message_col
        ].str.count(self.settings.word_regex)

        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat["n_words"] >= self.settings.min_words
        ]

        self.datafiles.chat["n_interpunction"] = self.datafiles.chat[
            self.settings.message_col
        ].str.count(self.settings.interpunction_regex)

        self.datafiles.chat[self.settings.norm_colnam] = (
            self.datafiles.chat["n_interpunction"] / self.datafiles.chat["n_words"]
        )

        self.datafiles.processed = self.datafiles.chat

    def get_p_values(
        self,
        datafiles: Optional[list[pd.Series | np.ndarray]] = None,
        norm_colnam: Optional[str] = None,
    ) -> tuple:
        norm_colnam = norm_colnam or self.settings.norm_colnam

        if not datafiles:
            if self.datafiles.processed is not None:
                datafiles = [
                    self.datafiles.processed.loc[
                        self.datafiles.processed["file"] == file
                    ][norm_colnam]
                    for file in self.datafiles.processed["file"].unique()
                ]
            else:
                raise ValueError(
                    f"No datafiles for p_values in {self.settings.file_stem}"
                )
        else:
            datafiles = [
                (
                    datafile[norm_colnam]
                    if isinstance(datafile, pd.DataFrame)
                    else datafile
                )
                for datafile in datafiles
            ]

        if len(datafiles) < 2:
            logger.warning(
                f"A p-value can't be calculated for {len(datafiles)} distributions. Returning p=1."
            )
            return 1, 1

        if len(datafiles) > 2:
            logger.warning(
                f"A p-value can't be calculated for {len(datafiles)} distributions. Returning p-value for the first 2."
            )

        u_stat, p_value_mw = stats.mannwhitneyu(
            datafiles[0], datafiles[1], alternative="two-sided"
        )

        ks_stat, p_value_ks = stats.ks_2samp(datafiles[0], datafiles[1])

        return p_value_mw, p_value_ks


class DistributionPlotter(BasePlot):
    settings: DistributionSettings

    def __init__(self, settings: DistributionSettings):
        super().__init__(settings)

    def plot(self, data: pd.DataFrame, **kwargs):
        super().create_figure(**kwargs)

        if "file" not in data.columns:
            files = ["dataset 1"]
        else:
            files = data.file.unique()

        for file in files:
            if file not in self.settings.input_settings:
                raise ValueError(
                    "Message file {file} not in input_settings (config.toml or keyword arguments passed to settings)"
                )

            file_settings = self.settings.input_settings[file]
            datafile = data.loc[data["file"] == file]

            self.ax.hist(
                datafile[self.settings.norm_colnam],
                density=True,
                alpha=self.settings.hist_alpha,
                color=file_settings["color"],
            )

            x, pdf = self.get_norm(
                datafile,
                self.settings.norm_colnam,
                file_col=file,
                norm_type=file_settings["type"],
            )

            norm_fig = self.ax.plot(
                x, pdf, file_settings["color"], linewidth=self.settings.linewidth
            )

            create_colored_annotation(
                self.ax,
                f"n = {len(datafile)}",
                norm_fig[0].get_xydata(),
                color=file_settings["color"],
                x_offset=self.settings.annotation_x_offset,
                y_offset=self.settings.annotation_y_offset,
            )

        plt.show()

    def get_norm(
        self,
        file: pd.DataFrame | pd.Series,
        norm_col: str,
        norm_type: str = "norm",
        file_col: Optional[str] = None,
    ) -> tuple:

        if isinstance(file, pd.DataFrame):
            if file_col:
                file = file.loc[file["file"] == file_col]

            file = file[norm_col]

        x = np.linspace(0, max(file) * 1.2, 1000)

        if norm_type == "norm":

            mu, sigma = stats.norm.fit(file)

            pdf = stats.norm.pdf(x, mu, sigma)

        elif norm_type == "halfnorm":

            loc, scale = stats.halfnorm.fit(file)

            pdf = stats.halfnorm.pdf(x, loc=loc, scale=scale)

        else:
            raise ValueError(
                f"norm_type must be norm or halfnorm for file {self.settings.file_stem}..."
            )

        return x, pdf


def main():

    settings = DistributionSettings(**AllVars())

    loader = DistributionLoader(settings)

    p_value_mw, p_value_ks = loader.get_p_values()

    plotter = DistributionPlotter(settings)

    plotter.plot(
        loader.datafiles.processed,
        p_value_mw=f"{p_value_mw:.3e}",
        p_value_ks=f"{p_value_ks:.3e}",
    )


if __name__ == "__main__":
    main()
