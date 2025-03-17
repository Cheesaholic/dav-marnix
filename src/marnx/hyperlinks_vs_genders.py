import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Local imports
from settings import MessageFileLoader, BasePlot, create_distribution, get_random_color, is_hyperlink, PlotSettings


class DistributionLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()
        super().get_datafiles()
        self.clean_transform_data()

    def clean_transform_data(self):

        self.datafiles[0]["gender"] = [self.datafiles[1][x] for x in self.datafiles[0]["author"]]

        self.datafiles[0] = self.datafiles[0].loc[self.datafiles[0]["author"].map(self.datafiles[0].groupby("author")["message"].count() >= self.min_messages) == True]

        self.datafiles[0]["has_hyperlink"] = self.datafiles[0]["message"].apply(is_hyperlink)

        agg_df = self.datafiles[0].groupby(["gender", "original_author"]).agg(
        total_messages=('message', 'count'),
        total_links=('has_hyperlink', 'sum')
        )

        agg_df["link_pct"] = (agg_df["total_links"] / agg_df["total_messages"]) * 100

        agg_df.sort_values("link_pct", inplace=True)

        genders = agg_df.index.get_level_values('gender').unique().to_list()

        gender_dict = {}

        for gender in genders:
            gender_dict[gender] = create_distribution(agg_df.loc[gender]["link_pct"].to_numpy())
        
        self.datafiles[0] = gender_dict



class DistributionPlotter(BasePlot):
    def __init__(self, settings):
        super().__init__(settings)
    
    def plot(self, loader):
        super().create_figure(loader=loader)

        for gender, gender_data in loader.datafiles[0].items():

            if hasattr(loader, "gender_colors") and gender in loader.gender_colors:
                gender_color = loader.gender_colors[gender]
            else:
                gender_color = get_random_color()
            
            self.ax.plot(gender_data[0][0], gender_data[0][1], color=gender_color, label=gender, linewidth=2)
            self.ax.plot(gender_data[1][0], gender_data[1][1], color=gender_color, linestyle="--", alpha=0.2, label= f"{gender} bell-curve", linewidth=2)
        
        
        # TODO: Create p-value for more than 2 genders
        p = round(ttest_ind(loader.datafiles[0]["M"][0][0], loader.datafiles[0]["F"][0][0]).pvalue, 6)

        self.fig.suptitle(loader.suptitle.format(p=p), fontsize=18)
        # TODO: Create title for more than 2 genders
        self.ax.set_title(loader.title.format(num_m = len(loader.datafiles[0]["M"][0][0]), num_f = len(loader.datafiles[0]["F"][0][0]), min_messages=loader.min_messages), fontsize=11, y=0.995)

        self.ax.axvline(color = "grey", alpha = 0.3)
        
        plt.show()

def main():

    loader = DistributionLoader()

    plotter = DistributionPlotter(PlotSettings())

    plotter.plot(loader)

if __name__ == "__main__":
    main()