from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap.umap_ import UMAP

# Local imports
from marn_x.settings import (AllVars, BasePlot, MessageFileLoader,
                             PlotSettings, remove_edited, remove_emoji,
                             remove_exclude_terms, remove_image,
                             remove_more_information, remove_numbers,
                             remove_removed, remove_security_code, remove_url)


class TopicSettings(PlotSettings):
    use_pretrained_model: bool
    pretrained_model_name: str
    exclude_terms: list[str]
    bert_sentence_model: str
    language: str
    sentence_regex: str
    sentence_min_token_len: int

    umap_neighbors: int
    umap_n_components: int
    umap_min_dist: float | int
    umap_metric: str
    umap_random_state: int

    hdbscan_min_cluster_size: int
    hdbscan_metric: str
    hdbscan_cluster_selection_method: str
    hdbscan_prediction_data: bool

    representation_model: str

    bertopic_top_n_words: int
    bertopic_verbose: bool

    plotly_width: int
    plotly_xref: str


class TopicLoader(MessageFileLoader):
    settings: TopicSettings

    def __init__(self, settings: TopicSettings):
        super().__init__(settings)

    def clean_transform_data(self):

        self.datafiles.chat[self.settings.message_col] = self.datafiles.chat[
            self.settings.message_col
        ].str.split(self.settings.sentence_regex)
        self.datafiles.chat = self.datafiles.chat.explode(self.settings.message_col)
        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat[self.settings.message_col].notna()
        ]

        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)
            .apply(remove_image)
            .apply(remove_emoji)
            .apply(remove_exclude_terms, exclude_terms=self.settings.exclude_terms)
            .apply(remove_numbers)
            .apply(remove_more_information)
            .apply(remove_security_code)
            .apply(remove_removed)
            .apply(remove_edited)
            .str.strip()
        )

        self.datafiles.chat = self.datafiles.chat.loc[
            (
                self.datafiles.chat[self.settings.message_col].str.count(" ")
                >= self.settings.sentence_min_token_len - 1
            )
        ]

        self.datafiles.chat.drop_duplicates(inplace=True, ignore_index=True)

    def generate_topics(self, df: Optional[pd.DataFrame] = None) -> BERTopic:

        df = df or self.datafiles.chat

        if df is None:
            raise ValueError(f"No Dataframe for {self.settings.file_stem}")

        # Extract embeddings
        embedding_model = SentenceTransformer(self.settings.bert_sentence_model)

        # Reduce dimensionality
        umap_model = UMAP(
            n_neighbors=self.settings.umap_neighbors,
            n_components=self.settings.umap_n_components,
            min_dist=self.settings.umap_min_dist,
            metric=self.settings.umap_metric,
            random_state=self.settings.umap_random_state,
        )

        # Cluster reduced embeddings
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.settings.hdbscan_min_cluster_size,
            metric=self.settings.hdbscan_metric,
            cluster_selection_method=self.settings.hdbscan_cluster_selection_method,
            prediction_data=self.settings.hdbscan_prediction_data,
        )

        # Tokenize topics
        if hasattr(self.datafiles, "stopwords"):
            vectorizer_model = CountVectorizer(
                stop_words=self.datafiles.stopwords[
                    self.datafiles.stopwords.columns[0]
                ].to_list()
            )
        else:
            vectorizer_model = CountVectorizer()

        # Create topic representation
        ctfidf_model = ClassTfidfTransformer()

        representation_model = PartOfSpeech(self.settings.representation_model)

        # Use the representation model in BERTopic on top of the default pipeline
        topic_model = BERTopic(representation_model=representation_model)

        topic_model = BERTopic(
            language=self.settings.language,
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            top_n_words=self.settings.bertopic_top_n_words,
            verbose=self.settings.bertopic_verbose,
        )

        topic_model.fit_transform(df[self.settings.message_col])

        self.datafiles.topic_model = topic_model

        return self.datafiles.topic_model

    def get_pretrained_model(self, model: Optional[str] = None) -> BERTopic:
        model = model or self.settings.pretrained_model_name

        try:
            models_path = (
                Path(__file__).parent / "../../" / self.settings.models
            ).resolve()
        except ValueError:
            logger.error("'models' path in config.toml does not exist.")

        full_path = models_path / model

        if not full_path.exists():
            raise FileNotFoundError(f"No model named {model} in models directory.")

        self.datafiles.topic_model = BERTopic.load(full_path)

        return self.datafiles.topic_model


class TopicPlotter(BasePlot):
    settings: TopicSettings

    def __init__(self, settings: TopicSettings):
        super().__init__(settings)

    def plot(self, topic_model: BERTopic, docs: list[str]):

        embeddings = SentenceTransformer(self.settings.bert_sentence_model).encode(docs)

        self.fig = topic_model.visualize_documents(docs, embeddings=embeddings)

        self.fig.update_layout(
            title=go.layout.Title(
                text=f"<sup>{self.settings.uber_suptitle}</sup><br><b>{self.settings.suptitle}</b><br><sup>{self.settings.title}</sup>".format(
                    bert_sentence_model=self.settings.bert_sentence_model
                ),
                xref=self.settings.plotly_xref,
                x=self.settings.title_x,
                y=self.settings.title_y,
            ),
            width=self.settings.plotly_width,
        )

        self.fig.show()


def main():

    settings = TopicSettings(**AllVars())

    loader = TopicLoader(settings)

    loader.clean_transform_data()

    if loader.settings.use_pretrained_model:

        loader.get_pretrained_model()

    else:

        loader.generate_topics()

    plotter = TopicPlotter(settings)

    plotter.plot(
        loader.datafiles.topic_model, loader.datafiles.chat[settings.message_col].values
    )

    logger.warning(
        f"If this plot is empty: Please run the Jupyter Notebook titled {settings.file_stem}. Something is wrong with this version of Plotly in combination with Bertopic. I have tried to debug, but unfortunately nothing has helped. The Notebook uses the same code as here!"
    )


if __name__ == "__main__":
    main()
