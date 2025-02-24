import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from GraphTopic.helpers.config import embeddingModel
from GraphTopic.helpers.topic_extraction import TopicManager, KeyPhrase
from GraphTopic.helpers.topic_merging import (
    get_merged_topics,
    get_topics_with_pruned_kps,
    KeyphrasesSelectionStrategy,
    flatten_graph_df,
    flatten_df_to_dict,
)
from GraphTopic.helpers.kps import getKeyPhrasesUsingKeyBERT

# type
from typing import List, Tuple, Optional, Callable


class GraphTopic:
    """
    GraphTopic: A class for extracting and merging topics from a list of documents using a graph-based approach.

    This class implements a graph-based topic extraction method. It first extracts keyphrases from each document and the merged document.
    Then, it builds an initial graph of trivial topics based on the keyphrases from the merged document.
    Next, it iteratively merges these trivial topics by processing each document and its keyphrases.
    Finally, it merges similar topics based on embedding similarity to generate the final set of topics.

    The main steps involved are:
    1. Keyphrase Extraction: Extract keyphrases from the merged document (for initial topic creation) and each individual document (for topic merging).
    2. Trivial Topic Initialization: Create initial topics based on the top keyphrases from the merged document.
    3. Latent Topic Extraction: Merge trivial topics by processing each document and its keyphrases, building a graph of related topics.
    4. Final Topic Merging: Merge semantically similar latent topics based on keyphrase embeddings to produce a concise set of final topics.

    The algorithm leverages keyphrase embeddings and graph structures to identify and merge related topics,
    allowing for a more nuanced and comprehensive topic representation compared to traditional methods.

    Args:
        docs (List[str]): List of document texts to analyze.
        TopM (int, optional):  Determines the granularity of initial trivial topics.
                                Higher TopM (↑) considers more keyphrases from the merged document, potentially leading to more granular and specific initial topics.
                                Defaults to 500.
        TopN (int, optional):  Controls the keyphrases extracted per document during latent topic extraction.
                                Lower TopN (↓) extracts fewer keyphrases per document, potentially leading to broader and more general latent topics.
                                Defaults to 10.
        keyphrase_extractor (Callable[[str, int], List[Tuple[str, float]]], optional):
            Function to extract keyphrases from text.
            It should take text and the number of keyphrases to extract as input and return a list of tuples,
            where each tuple contains a keyphrase (str) and its weight (float).
            Defaults to `getKeyPhrasesUsingKeyBERT`.
            Example:
            ```python
            def getKeyPhrases(text, n):
                # ... keyphrase extraction logic ...
                return [("keyphrase1", 0.9), ("keyphrase2", 0.8), ...]
            ```
        embedding_model (SentenceTransformer, optional): SentenceTransformer model for embedding keyphrases during topic merging.
                                                        Defaults to `embeddingModel` from `GraphTopic.helpers.config`.
        pretty_print (bool, optional): Enables pretty printing of intermediate steps for debugging and visualization.
                                       Defaults to False.
        prune_node_count_less_than (int, optional):  Minimum number of nodes (keyphrases) a topic must have to be retained in the latent topic graph.
                                                       Topics with fewer nodes are pruned, helping to remove very specific or less relevant topics.
                                                       Defaults to 2.
        overlapping_threshold (float, optional):  Threshold for merging topics based on embedding similarity during final topic merging.
                                                  A higher threshold means topics need to be more similar to be merged, resulting in more, potentially more specific, final topics.
                                                  Defaults to 0.5.
        max_topics (float, optional): Maximum number of final topics to retrieve. If 'inf', all topics are returned.
                                       Defaults to float("inf").
        max_keyphrases_per_topic (int, optional): Maximum number of keyphrases to include in each final topic.
                                                     Defaults to 10**2 (100).
        keyphrases_selection_strategy (KeyphrasesSelectionStrategy, optional):
            Strategy for selecting top keyphrases to represent each final topic when `max_topics` is specified.
            - KeyphrasesSelectionStrategy.IMP_NODE (default): Selects keyphrases based on outdegree node centrality in the topic graph (most important nodes).
            - KeyphrasesSelectionStrategy.CDIST: Selects keyphrases based on cosine similarity to the topic centroid embedding (closest to the topic center).
            - KeyphrasesSelectionStrategy.RNDOM: Selects keyphrases randomly.
            Defaults to KeyphrasesSelectionStrategy.CDIST.

    Usage:
        ```python
        from GraphTopic import GraphTopic

        documents = [
            "Document 1 text...",
            "Document 2 text...",
            # ... more documents
        ]

        graph_topic_model = GraphTopic(
            docs=documents,
            TopM=300,
            TopN=8,
            overlapping_threshold=0.6,
        )

        topics = graph_topic_model.get_topics()
    """

    def __init__(
        self,
        docs: List[str],
        TopM: int = 500,
        TopN: int = 10,
        keyphrase_extractor: Callable[
            [str, int], List[Tuple[str, float]]
        ] = getKeyPhrasesUsingKeyBERT,
        embedding_model: SentenceTransformer = embeddingModel,
        pretty_print=False,
        prune_node_count_less_than=2,
        overlapping_threshold=0.5,
        max_topics: float = float("inf"),
        max_keyphrases_per_topic: int = 10**2,
        keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST,
    ):
        """
        Initializes the GraphTopic model with documents and configurations.

        Args:
            docs (List[str]): List of document texts.
            TopM (int): Top M keyphrases to use for merging topics.
            TopN (int): Top N keyphrases to extract from each document for latent topic extraction.
            keyphrase_extractor (callable): Keyphrase extraction function.
                def getKeyPhrases(text,n):
                    return [(kp1,weight),(kp2,weight)]
            prune_node_count_less_than (int): Number of nodes to prune from the graph.
            pretty_print (bool): Flag to enable/disable pretty printing of merging steps.
            overlapping_threshold (float): Threshold for merging topics based on embedding similarity.
            max_keyphrases_per_topic (int): Maximum keyphrases per topic in the final output.
            max_topics (int): Maximum number of topics to retrieve in the final output.
            keyphrases_selection_strategy (KeyphrasesSelectionStrategy):
                Strategy for selecting top keyphrases from each topic:
                'IMP_NODE'(default;based on outdegree node centrality),
                'CDIST' (cosine similarity to topic centroid),
                'RNDOM'(random).
        """
        self._TopM = TopM
        self._TopN = TopN
        self._docs = docs
        self._pretty_print = pretty_print
        self._keyphrase_extractor = keyphrase_extractor
        self._overlapping_threshold = overlapping_threshold
        self._max_topics = max_topics
        self._max_keyphrases_per_topic = max_keyphrases_per_topic
        self._keyphrases_selection_strategy = keyphrases_selection_strategy
        self._prune_node_count_less_than = prune_node_count_less_than
        self._embedding_model = embeddingModel
        self._final_graph_df = None
        self._final_flatten_df = None
        self.topics: List[List] = []

        if self._keyphrase_extractor is None:
            raise Exception("Please provide a keyphrase extractor function")

        # !get merged kps from merged docs
        self._merged_kps: List[str] = self.__extract_kps_merged_from_merged_doc()
        self.__extract_latent_topics()

    def __get_kps(
        self,
        text: str,
        n: int,
        keyphrase_extractor: Callable[[str, int], List[Tuple[str, float]]],
    ) -> List[str]:
        """
        Extracts keyphrases from the given text using the provided keyphrase extractor function.

        Args:
            text (str): Input text from which to extract keyphrases.
            n (int): Number of keyphrases to extract.
            keyphrase_extractor(text, n): Function that takes text and n and returns a list of keyphrase tuples (keyphrase, weight).

        Returns:
            List[str]: List of extracted keyphrases (keyphrases only, weights are discarded).
        """
        kps = keyphrase_extractor(text, n)
        return [kp for kp, _ in kps]

    def __extract_kps_merged_from_merged_doc(self) -> List[str]:
        """
        Extracts the top M keyphrases from the merged document.

        The merged document is created by joining all input documents with ". " separator.
        These keyphrases are used to initialize trivial topics in the latent topic extraction phase.

        Returns:
            List[str]: List of keyphrases extracted from the merged document.
        """
        merged_doc = ". ".join(self._docs)
        # if len(merged_doc) >= 1_000_000:
        #     raise Exception(
        #         "Merged document length is 1_000_000. Please increase the length of the documents."
        #     )
        print(f"> Extracting top M:{self._TopM} keyphrases from merged documents...")
        extracted = self.__get_kps(
            merged_doc,
            n=self._TopM,
            keyphrase_extractor=self._keyphrase_extractor,
        )
        print(f"Total {len(extracted)} keyphrases extracted\n")
        return extracted

    def __extract_latent_topics(self) -> None:
        """
        Extracts latent topics from the documents using a graph-based merging approach.

        This method performs the following steps:
        1. Initializes trivial topics: Creates initial topics based on the top M keyphrases extracted from the merged document.
           Each keyphrase becomes a trivial topic. Only multi-word keyphrases are considered to avoid single-word topics.
        2. Merges subtopics: Iterates through each document and its top N keyphrases. For each document's keyphrases, it merges them into existing trivial topics
           if they are semantically related. This builds a graph where topics are connected if they share keyphrases from the same document.
        3. Prunes topics: Removes topics with a node count (number of associated keyphrases) less than `self._prune_node_count_less_than`.
           This step filters out very specific or less significant topics.
        4. Generates latent topic DataFrame: Creates a Pandas DataFrame representing the latent topic graph.
        5. Flattens DataFrame:  Transforms the graph DataFrame into a flattened DataFrame and dictionary for easier access and processing.

        The `self.latent_topics` attribute is updated with a list of dictionaries, where each dictionary represents a latent topic
        and contains its keyphrases and related information.

        Returns:
            None
        """
        try:
            d = TopicManager()

            # building trivial topics
            print(
                f"> Initializing {len(self._merged_kps)} trivial topics from top M keyphrases..."
            )
            for kp in self._merged_kps:
                # Only Considering Phrase, single words are avoid [as we are using phrase-bert]
                if len(str(kp).split(" ")) < 2:
                    continue
                d.add_new_topic_from_keyphrases([KeyPhrase(paper_id="", keyphrase=kp)])

            print("> Extracting latent topics...")
            for doc in tqdm(self._docs):
                kps = self.__get_kps(
                    doc,
                    n=self._TopN,
                    keyphrase_extractor=self._keyphrase_extractor,
                )
                kps_record = []
                for kp in kps:
                    kps_record.append(KeyPhrase(paper_id="", keyphrase=kp))
                d.merge_subtopics(kps_record)

            d.clear_topic_with_node_count_n(self._prune_node_count_less_than)
            latent_graph_df = d.generate_df()

            flatten_latent_df = flatten_graph_df(latent_graph_df)
            latent_topics_dicts = flatten_df_to_dict(flatten_latent_df)

            print(f"Total {len(latent_topics_dicts)} latent subtopics generated\n")

            self._latent_graph_df = latent_graph_df
            self._flatten_latent_df = flatten_latent_df
            self.latent_topics = latent_topics_dicts

        except Exception as e:
            print(e)

    def __extract_final_topics(
        self, th
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[List]]]:
        """
        Merges semantically similar latent topics to generate the final set of topics.

        This method uses keyphrase embeddings and a cosine similarity threshold (`th`) to merge latent topics.
        Topics with an embedding similarity greater than or equal to `th` are merged into a single final topic.

        Args:
            th (float): Cosine similarity threshold for merging latent topics.

        Returns:
            Optional[Tuple[pd.DataFrame, pd.DataFrame, List[List]]]:
                A tuple containing:
                - final_graph_df (pd.DataFrame): DataFrame representing the final topic graph after merging.
                - final_flatten_df (pd.DataFrame): Flattened DataFrame of the final topic graph.
                - _topics (List[List]): List of final topics, where each topic is a list of keyphrases.
                Returns None if latent topics have not been extracted yet or if an error occurs.
        """

        print("> Merging subtopics using embeddings...")
        if (
            self._latent_graph_df is None
            or self._latent_graph_df.shape[0] == 0  # type: ignore
            or self._flatten_latent_df is None
        ):
            raise Exception("Please extract latent topics first")
        try:
            final_graph_df, final_flatten_df, _topics = get_merged_topics(
                self._latent_graph_df,  # type: ignore
                self._embedding_model,
                th,
                self._pretty_print,
            )
            print(f"Final {len(_topics)} topics generated\n")
            return final_graph_df, final_flatten_df, _topics

        except Exception as e:
            print(e)

    def get_topics(self) -> List[List]:
        """
        Retrieves the final topics extracted by the GraphTopic model.

        This method orchestrates the final topic extraction process by calling `__extract_final_topics`
        with the `overlapping_threshold` as the similarity threshold.

        It also handles the case where `max_topics` is specified, pruning the final topics to the desired number
        using the specified `keyphrases_selection_strategy`. If `max_topics` is 'inf', all extracted topics are returned.

        Returns:
            List[List]: List of final topics.
                          If `max_topics` is 'inf', each topic is a list of all keyphrases belonging to that topic.
                          If `max_topics` is a finite number, each topic is a list of top keyphrases (up to `max_keyphrases_per_topic`)
                          selected according to the `keyphrases_selection_strategy`.
        """
        (
            self._final_graph_df,
            self._final_flatten_df,
            self.topics,  # final topics are saved in this var.
        ) = self.__extract_final_topics(th=self._overlapping_threshold)  # type: ignore

        if self._max_topics == float("inf"):
            return [kps[: self._max_keyphrases_per_topic] for kps in self.topics]

        return get_topics_with_pruned_kps(
            flatten_df=self._final_flatten_df,
            embeddingModel=self._embedding_model,
            max_keyphrases_per_topic=self._max_keyphrases_per_topic,
            keyphrases_selection_strategy=self._keyphrases_selection_strategy,
            max_topics=self._max_topics,
        )  # type: ignore

    def get_latent_graph_df(self):
        """
        Returns the Pandas DataFrame representing the latent topic graph.

        This DataFrame contains information about the latent topics extracted before the final merging step.
        It can be used for further analysis and visualization of the intermediate topic structure.

        Returns:
            pd.DataFrame: DataFrame representing the latent topic graph.
                          The DataFrame structure typically includes columns for topic IDs, keyphrases, node counts, and potentially other graph-related metrics.
        """
        return self._latent_graph_df
