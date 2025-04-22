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

from marn_x.utils.data_transformers import (MessageFileLoader, remove_edited,
                                            remove_emoji, remove_exclude_terms,
                                            remove_image,
                                            remove_more_information,
                                            remove_numbers, remove_removed,
                                            remove_security_code, remove_url)
# Local imports
from marn_x.utils.settings import AllVars, BasePlot, PlotSettings


class TopicSettings(PlotSettings):
    """
    Configuration settings for topic modeling and visualization.
    Extends PlotSettings with specific parameters for topic modeling workflow.
    """

    # Topic label generation settings
    labels_nr_words: int  # Number of words to include in each topic label
    labels_prefix: bool  # Whether to prefix labels with topic numbers
    labels_word_length: bool  # Whether to display word lengths in labels
    labels_separator: str  # Separator between words in topic labels
    labels_aspect: (
        str  # Aspect to emphasize in topic representation (e.g., "word", "pos")
    )

    # Model selection settings
    use_pretrained_model: (
        bool  # Whether to use a saved model instead of training a new one
    )
    pretrained_model_name: str  # Name of the pretrained model file to load
    exclude_terms: list[str]  # Terms to exclude from text analysis
    bert_sentence_model: str  # Name of the sentence transformer model to use
    language: str  # Language of the text data
    sentence_regex: str  # Regex pattern to split text into sentences
    sentence_min_token_len: (
        int  # Minimum number of tokens a sentence must have to be included
    )

    # UMAP dimensionality reduction settings
    umap_neighbors: int  # Number of neighbors for local neighborhood approximation
    umap_n_components: int  # Target dimension for the reduced space
    umap_min_dist: float | int  # Minimum distance between points in the embedding space
    umap_metric: str  # Distance metric for UMAP
    umap_random_state: int  # Random seed for reproducibility

    # HDBSCAN clustering settings
    hdbscan_min_cluster_size: int  # Minimum size of clusters
    hdbscan_metric: str  # Distance metric for clustering
    hdbscan_cluster_selection_method: str  # Method to select clusters ('eom' or 'leaf')
    hdbscan_prediction_data: bool  # Whether to store prediction data for later use

    # Representation model settings
    representation_model: (
        str  # Model used for topic representation (e.g., "noun_chunks")
    )

    # BERTopic settings
    bertopic_top_n_words: int  # Number of words to include in topic representation
    bertopic_verbose: bool  # Whether to print progress information

    # Visualization settings
    plotly_width: int  # Width of plotly visualizations
    plotly_xref: str  # X-reference for plot title positioning


class TopicLoader(MessageFileLoader):
    """
    Handles data loading, preprocessing and topic model creation.
    Extends MessageFileLoader with specific methods for topic modeling.
    """

    settings: TopicSettings

    def __init__(self, settings: TopicSettings):
        """Initialize the topic loader with the provided settings."""
        super().__init__(settings)

    def clean_transform_data(self):
        """
        Preprocess and clean the loaded text data:
        1. Split messages into sentences
        2. Apply multiple text cleaning functions
        3. Filter out short sentences
        4. Remove duplicates
        """
        # Merge all data files into a single dataframe
        self.datafiles.chat = self.datafiles.merge()

        # Split each message into sentences based on regex pattern
        self.datafiles.chat[self.settings.message_col] = self.datafiles.chat[
            self.settings.message_col
        ].str.split(self.settings.sentence_regex)

        # Explode the dataframe so each sentence gets its own row
        self.datafiles.chat = self.datafiles.chat.explode(self.settings.message_col)

        # Remove rows with NaN values in the message column
        self.datafiles.chat = self.datafiles.chat.loc[
            self.datafiles.chat[self.settings.message_col].notna()
        ]

        # Apply a series of text cleaning functions to each message
        self.datafiles.chat[self.settings.message_col] = (
            self.datafiles.chat[self.settings.message_col]
            .apply(remove_url)  # Remove URLs
            .apply(remove_image)  # Remove image references
            .apply(remove_emoji)  # Remove emojis
            .apply(
                remove_exclude_terms, exclude_terms=self.settings.exclude_terms
            )  # Remove specified terms
            .apply(remove_numbers)  # Remove numbers
            .apply(remove_more_information)  # Remove "more information" sections
            .apply(remove_security_code)  # Remove security codes
            .apply(remove_removed)  # Remove "removed" placeholders
            .apply(remove_edited)  # Remove "edited" indicators
            .str.strip()  # Remove leading/trailing whitespace
        )

        # Filter out sentences that are too short (based on token count)
        self.datafiles.chat = self.datafiles.chat.loc[
            (
                self.datafiles.chat[self.settings.message_col].str.count(" ")
                >= self.settings.sentence_min_token_len - 1
            )
        ]

        # Remove duplicate sentences
        self.datafiles.chat.drop_duplicates(inplace=True, ignore_index=True)

    def generate_topics(self, df: Optional[pd.DataFrame] = None) -> BERTopic:
        """
        Create and train a new BERTopic model with the preprocessed data.

        Args:
            df: Optional dataframe to use instead of the internally stored data

        Returns:
            The trained BERTopic model

        Raises:
            ValueError: If no dataframe is available
        """
        # Use provided dataframe or default to internal data
        df = df or self.datafiles.chat

        if df is None:
            raise ValueError(f"No Dataframe for {self.settings.file_stem}")

        logger.info("Creating new Bertopic model. This is going to take a while...")

        # Create sentence embedding model to convert text to vectors
        embedding_model = SentenceTransformer(self.settings.bert_sentence_model)

        # Set up UMAP for dimensionality reduction of embeddings
        umap_model = UMAP(
            n_neighbors=self.settings.umap_neighbors,  # Balance between local and global structure
            n_components=self.settings.umap_n_components,  # Output dimensions (typically 2 for visualization)
            min_dist=self.settings.umap_min_dist,  # Minimum distance between points in embedding
            metric=self.settings.umap_metric,  # Distance metric
            random_state=self.settings.umap_random_state,  # For reproducibility
        )

        # Set up HDBSCAN for clustering the reduced embeddings
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.settings.hdbscan_min_cluster_size,  # Min size for a group to be considered a cluster
            metric=self.settings.hdbscan_metric,  # Distance metric for clustering
            cluster_selection_method=self.settings.hdbscan_cluster_selection_method,  # Algorithm for selecting flat clusters
            prediction_data=self.settings.hdbscan_prediction_data,  # Store data for predicting cluster membership
        )

        # Set up vectorizer for tokenizing the text
        # Use custom stopwords if available, otherwise use default
        if hasattr(self.datafiles, "stopwords"):
            vectorizer_model = CountVectorizer(
                stop_words=self.datafiles.stopwords[
                    self.datafiles.stopwords.columns[0]
                ].to_list()
            )
        else:
            vectorizer_model = CountVectorizer()

        # Set up c-TF-IDF model for topic representation
        ctfidf_model = ClassTfidfTransformer()

        # Create the topic representation model (e.g., based on parts of speech)
        representation_model = PartOfSpeech(self.settings.representation_model)

        # Initialize BERTopic with all components and settings
        topic_model = BERTopic(
            language=self.settings.language,  # Language for text processing
            embedding_model=embedding_model,  # Model for creating embeddings
            umap_model=umap_model,  # Dimensionality reduction model
            hdbscan_model=hdbscan_model,  # Clustering model
            vectorizer_model=vectorizer_model,  # Text vectorization model
            ctfidf_model=ctfidf_model,  # TF-IDF transformation model
            representation_model=representation_model,  # Topic representation model
            top_n_words=self.settings.bertopic_top_n_words,  # Number of words per topic
            verbose=self.settings.bertopic_verbose,  # Print progress information
        )

        # Fit the model to the data and transform the documents
        topic_model.fit_transform(df[self.settings.message_col])

        # Generate human-readable topic labels
        topic_labels = topic_model.generate_topic_labels(
            nr_words=self.settings.labels_nr_words,  # Number of words per topic label
            topic_prefix=self.settings.labels_prefix,  # Whether to include topic number
            word_length=self.settings.labels_word_length,  # Whether to show word lengths
            separator=self.settings.labels_separator,  # Word separator in labels
            aspect=(  # Aspect to emphasize
                None
                if self.settings.labels_aspect == ""
                else self.settings.labels_aspect
            ),
        )

        # Apply the generated labels to the model
        topic_model.set_topic_labels(topic_labels)

        # Store the model for later use
        self.datafiles.topic_model = topic_model

        return self.datafiles.topic_model

    def get_pretrained_model(self, model: Optional[str] = None) -> BERTopic:
        """
        Load a pretrained BERTopic model from disk.

        Args:
            model: Optional model name to override the one in settings

        Returns:
            The loaded BERTopic model

        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        model = model or self.settings.pretrained_model_name

        try:
            # Resolve the path to the models directory
            models_path = (
                Path(__file__).parent / "../../" / self.settings.models
            ).resolve()
        except ValueError:
            logger.error("'models' path in config.toml does not exist.")

        # Full path to the specified model
        full_path = models_path / model

        if not full_path.exists():
            raise FileNotFoundError(
                f"No model named {model} in {self.settings.models} directory."
            )

        logger.info(f"Retrieving model {model}. This could take some time...")

        # Load the model and store it
        self.datafiles.topic_model = BERTopic.load(full_path)

        return self.datafiles.topic_model


class TopicPlotter(BasePlot):
    """
    Handles visualization of topic models using Plotly.
    Extends BasePlot with specific methods for topic visualization.
    """

    settings: TopicSettings

    def __init__(self, settings: TopicSettings):
        """Initialize the topic plotter with the provided settings."""
        super().__init__(settings)

    def plot(self, topic_model: BERTopic, docs: list[str]):
        """
        Create and display a visualization of the topic model.

        Args:
            topic_model: The trained BERTopic model
            docs: List of documents/sentences to visualize
        """
        # Generate embeddings for the documents
        embeddings = SentenceTransformer(self.settings.bert_sentence_model).encode(docs)

        # Create visualization of documents with their topic assignments
        self.fig = topic_model.visualize_documents(
            docs, embeddings=embeddings, custom_labels=True
        )

        # Update the layout of the visualization
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
            showlegend=False if self.settings.hide_legend else True,
        )

        # Display the visualization
        self.fig.show()

        # Save the visualization to a HTML file
        self.to_html()


def main():
    """
    Main function to run the topic modeling workflow:
    1. Load settings
    2. Load and preprocess data
    3. Create or load a topic model
    4. Create and display visualizations
    """
    # Load all settings from configuration
    settings = TopicSettings(**AllVars())

    # Initialize the topic loader with settings
    loader = TopicLoader(settings)

    # Clean and preprocess the data
    loader.clean_transform_data()

    # Either load a pretrained model or train a new one
    if loader.settings.use_pretrained_model:
        loader.get_pretrained_model()
    else:
        loader.generate_topics()

    # Initialize the topic plotter with settings
    plotter = TopicPlotter(settings)

    # Create and display topic visualization
    plotter.plot(
        loader.datafiles.topic_model, loader.datafiles.chat[settings.message_col].values
    )


if __name__ == "__main__":
    main()
