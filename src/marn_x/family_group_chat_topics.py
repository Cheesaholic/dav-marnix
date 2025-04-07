from pathlib import Path
from typing import Optional

from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

import plotly.graph_objects as go

# Local imports
from settings import (
    MessageFileLoader, 
    PlotSettings, 
    BasePlot, 
    remove_url,
    remove_emoji,
    remove_image,
    remove_family_names,
    remove_numbers
)


class TopicLoader(MessageFileLoader):
    def __init__(self):
        super().__init__()

    def clean_transform_data(self):

        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col].str.split(self.sentence_regex)
        self.datafiles.chat = self.datafiles.chat.explode(self.message_col + "_split")
        self.datafiles.chat = self.datafiles.chat.loc[self.datafiles.chat[self.message_col + "_split"].notna()]

        self.datafiles.chat[self.message_col + "_split"] = self.datafiles.chat[self.message_col + "_split"].apply(remove_url) \
                                                                                                        .apply(remove_image) \
                                                                                                        .apply(remove_emoji) \
                                                                                                        .apply(remove_family_names, family_names=self.family_names) \
                                                                                                        .apply(remove_numbers) \
                                                                                                        .str.strip()
        
        self.datafiles.chat = self.datafiles.chat.loc[(self.datafiles.chat[self.message_col + "_split"].str.count(' ') >= self.sentence_min_token_len -1) & \
                                                      ~(self.datafiles.chat[self.message_col + "_split"].str.contains("Je beveiligingscode voor"))]
        
        self.datafiles.chat.drop_duplicates(inplace=True, ignore_index=True)

        self.datafiles.chat.to_parquet("lechat.parq")
    
    def generate_topics(self):

        # Extract embeddings
        embedding_model = SentenceTransformer(self.bert_sentence_model)

        print(self.umap_neighbors)

        # Reduce dimensionality
        umap_model = UMAP(n_neighbors=self.umap_neighbors, n_components=self.umap_n_components, min_dist=self.umap_min_dist, metric=self.umap_metric, random_state=self.umap_random_state)

        # Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=self.hdbscan_min_cluster_size, metric=self.hdbscan_metric, cluster_selection_method=self.hdbscan_cluster_selection_method, prediction_data=self.hdbscan_prediction_data)

        # Tokenize topics
        if hasattr(self.datafiles, "stopwords"):
            vectorizer_model = CountVectorizer(stop_words=self.datafiles.stopwords[self.datafiles.stopwords.columns[0]].to_list())
        else:
            vectorizer_model = CountVectorizer()

        # Create topic representation
        ctfidf_model = ClassTfidfTransformer()

        topic_model = BERTopic(
        language=self.language,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=self.bertopic_top_n_words,
        verbose=self.bertopic_verbose
        )

        topic_model.fit_transform(self.datafiles.chat[self.message_col + "_split"])

        self.datafiles.topic_model = topic_model

    def get_pretrained_model(self, model: Optional[str] = None):
        if model is None:
            model = self.pretrained_model_name
        try:
            models_path = Path("../../" + self.models).resolve()
        except:
            raise ValueError("'models' path in config.toml does not exist.")

        full_path = models_path / model

        if not full_path.exists():
            raise FileNotFoundError(f"No model named {model} in models directory.")
        
        self.datafiles.topic_model = BERTopic.load(full_path)


class TopicPlotter(BasePlot):
    def __init__(self, settings):
        super().__init__(settings)
    
    def plot(self, loader):
        # super().create_figure(loader=loader)

        self.fig = loader.datafiles.topic_model.visualize_topics(width=1300)

        self.fig.update_layout(
            title=go.layout.Title(
                text= f"<b>{loader.suptitle}</b><br><sup>{loader.title.format(bert_sentence_model = loader.bert_sentence_model)}</sup>",
                xref="paper",
                x=0.5
            )
        )

        self.fig.show()


def main():

    loader = TopicLoader()

    if loader.use_pretrained_model == True:
        loader.get_pretrained_model()
    else:
        loader.clean_transform_data()
        loader.generate_topics()

    plotter = TopicPlotter(PlotSettings())

    plotter.plot(loader)

if __name__ == "__main__":
    main()