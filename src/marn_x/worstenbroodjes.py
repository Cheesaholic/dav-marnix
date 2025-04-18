from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local import
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, create_colored_annotation,
                             create_fourier_wave, fourier_model)


class WorstenbroodjesSettings(PlotSettings):
    api_data_col: str
    api_dt_format: str
    holidays: list[dict]
    aggregation_frequency: str
    fourier_components: int
    fourier_color: str
    fourier_alpha: float
    fourier_col: str
    data_color: str


class WorstenbroodjesLoader(MessageFileLoader):
    settings: WorstenbroodjesSettings

    def __init__(self, settings: WorstenbroodjesSettings):
        super().__init__(settings)

    def clean_transform_data(self):

        df = self.datafiles.wikipedia_api

        df[self.settings.timestamp_col] = pd.to_datetime(
            df[self.settings.timestamp_col], format=self.settings.api_dt_format
        )

        agg = (
            df.groupby(
                [
                    pd.Grouper(
                        key=self.settings.timestamp_col,
                        freq=self.settings.aggregation_frequency,
                    )
                ]
            )[self.settings.api_data_col]
            .sum()
            .reset_index()
            .sort_values(self.settings.timestamp_col)
            .set_index(self.settings.timestamp_col)
        )

        agg[self.settings.api_data_col] = agg[self.settings.api_data_col].astype(float)
        agg[self.settings.api_data_col + "_nrm"] = (
            agg[self.settings.api_data_col] - agg[self.settings.api_data_col].mean()
        )

        self.datafiles.processed = agg

    def create_fourier_model(
        self,
        custom_df: Optional[pd.DataFrame] = None,
        data_col: Optional[str] = None,
        fourier_components: Optional[int] = None,
        fourier_col: Optional[str] = None,
    ) -> np.ndarray:
        data = custom_df or self.datafiles.processed
        data_col = data_col or self.settings.api_data_col
        fourier_components = fourier_components or self.settings.fourier_components
        fourier_col = fourier_col or self.settings.fourier_col

        if data is None:
            raise ValueError(
                f"No DataFrame to create fourier model for {self.settings.file_stem}"
            )

        parameters = fourier_model(data[data_col + "_nrm"], fourier_components)

        y = create_fourier_wave(parameters)

        y += data[data_col].mean()

        if not custom_df and self.datafiles.processed is not None:
            self.datafiles.processed[fourier_col] = y

        return y


class WorstenbroodjesPlotter(BasePlot):
    settings: WorstenbroodjesSettings

    def __init__(self, settings: WorstenbroodjesSettings):
        super().__init__(settings)

    def plot(self, data, **kwargs):
        super().get_figure(**kwargs)

        ann_plt = self.ax.plot(
            data.index,
            data[self.settings.api_data_col],
            linewidth=self.settings.linewidth,
            color=self.settings.data_color,
        )
        self.ax.plot(
            data.index,
            data[self.settings.fourier_col],
            linestyle="--",
            linewidth=self.settings.linewidth,
            color=self.settings.fourier_color,
            alpha=self.settings.fourier_alpha,
        )

        for instance in self.settings.holidays:
            self.ax.axvspan(
                instance["start"],
                instance["end"],
                facecolor=instance["color"],
                alpha=instance["alpha"],
            )

        # # Add annotations
        create_colored_annotation(
            self.ax,
            self.settings.ylabel,
            ann_plt[0].get_xydata(),
            x_y="x",
            color=self.settings.data_color,
            x_offset=self.settings.annotation_x_offset,
            y_offset=self.settings.annotation_y_offset,
            fontweight="normal",
            ha="left",
        )

        plt.tight_layout()
        plt.show()


def main():

    settings = WorstenbroodjesSettings(**AllVars())

    loader = WorstenbroodjesLoader(settings)
    loader.clean_transform_data()
    loader.create_fourier_model()

    plotter = WorstenbroodjesPlotter(settings)

    plotter.plot(
        loader.datafiles.processed,
        ymax=loader.datafiles.processed[settings.api_data_col].max(),
    )


if __name__ == "__main__":
    main()
