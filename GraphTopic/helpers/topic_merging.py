# print("⌛ Importing libraries...")
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from rich.console import Console
from rich.table import Table
from enum import Enum

# from helpers.config import reducer, th
from .config import reducer, SentenceTransformer

from scipy.integrate import quad
import itertools
import random

# Set the seed to ensure reproducibility
seed_value = 42
random.seed(seed_value)


# Projecting the data points onto the line
def projection_onto_line(data_points, a, b):
    # Finding the unit vector along the line
    line_vector = b - a
    unit_vector = line_vector / np.linalg.norm(line_vector)
    unit_vector = unit_vector[np.newaxis, :]

    # Calculating the projection of each data point onto the line
    projections = np.dot(data_points - a, unit_vector.T) * unit_vector + a
    return projections


def pdf(data, mean: float, variance: float):
    # A normal continuous random variable.
    s1 = 1 / (np.sqrt(2 * np.pi * variance))
    s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
    return s1 * s2


def merge_indices(indices_list):
    """
    INPUT: (list of to be overlapped pairs:)
     [(0, 1), (0, 2),(3, 4), (3, 5),(7,8)]
    OUTPUT:
     [[0, 1, 2], [3, 4, 5], [7, 8]]

     Explanation:
     > (0,1) -> 0 to be merged with 1
     > (0,2) -> 0 to be merged with 2

     that means 0,1 and 2 should be merged together
     so the output is (0,1,2)

    """
    # Create an empty list to hold the merged indices.
    merged_list = []

    # Loop through the indices in the given list.
    for indices in indices_list:
        merged = False
        # Loop through the existing clusters in the merged list.
        for i, cluster in enumerate(merged_list):
            # If any of the indices in the current indices are already
            # in the cluster, add the remaining indices to the cluster.
            if any(idx in cluster for idx in indices):
                cluster.extend(set(indices) - set(cluster))
                merged = True
                break
        # If the indices were not merged with any existing clusters,
        # create a new cluster with the given indices.
        if not merged:
            merged_list.append(list(indices))

    # Return the merged list of indices.
    return merged_list


def get_overlap(indices, X_df):
    idx1, idx2 = indices
    X_val = X_df[X_df["topic"].isin(indices)].drop(columns=["topic"])
    # reduction= PCA(n_components=min(X_val.shape[0], X_val.shape[1],10)-1, random_state=45)
    # min(X_val.shape[0], X_val.shape[1],30)-1
    X_reduced = reducer.fit_transform(X_val)
    X_reduced = pd.DataFrame(X_reduced)
    X_reduced["topic"] = X_df[X_df["topic"].isin(indices)]["topic"].tolist()
    C1 = X_reduced[X_reduced["topic"] == idx1].drop(columns=["topic"]).to_numpy()
    C2 = X_reduced[X_reduced["topic"] == idx2].drop(columns=["topic"]).to_numpy()
    c1 = np.mean(C1, axis=0)
    c2 = np.mean(C2, axis=0)
    # Defining the line passing through the center points
    a = c1
    b = c2
    # Projecting the data points onto the line
    C1_projections = projection_onto_line(C1, a, b)
    C2_projections = projection_onto_line(C2, a, b)
    # left most point
    pl = np.minimum(C1_projections.min(axis=0), C2_projections.min(axis=0))
    PC1 = np.zeros((C1.shape[0],))
    for i, p in enumerate(C1_projections):
        PC1[i] = np.linalg.norm(p - pl)
    PC2 = np.zeros((C2.shape[0],))
    for i, p in enumerate(C2_projections):
        PC2[i] = np.linalg.norm(p - pl)

    overlap, _ = quad(
        lambda x: np.minimum(
            pdf(x, PC1.mean(), PC1.var()), pdf(x, PC2.mean(), PC2.var())
        ),
        -np.inf,
        np.inf,
    )
    return overlap


def merge_clusters(topic_pair_combinations, th, embedding_df, flatten_df):
    """
    INPUT: (list of pairs:)
     [(0, 1), (0, 2),(3, 4), (3, 5),(7,8)]
    OUTPUT:
     New embedding_df, flatten_df if overlapping threshold met or None,None
    """

    # [(0, 1), (0, 2), (0, 3),....
    similar_pair_list = []
    for each_pair in topic_pair_combinations:
        overlap = get_overlap(each_pair, embedding_df)

        # print(each_pair,round(overlap,2),f"th={round(th,2)}")
        if overlap >= th:
            similar_pair_list.append(each_pair)
    merged_list = merge_indices(similar_pair_list)  # [[0, 1, 2], [3, 4, 5]]
    if len(merged_list) < 1:
        return None, None
    else:
        for i, mergedIndices in enumerate(merged_list):
            embedding_df.loc[embedding_df["topic"].isin(mergedIndices), "topic"] = (
                mergedIndices[0]
            )
            flatten_df.loc[flatten_df["topic"].isin(mergedIndices), "topic"] = (
                mergedIndices[0]
            )
            return embedding_df, flatten_df


def recursive_merge_clusters(embedding_df, flatten_df, th):
    """
    Recursively merge clusters until no more merging is possible, and return the final merged clusters.
    """
    # global th
    # global counter
    # th = th + counter * alpha
    # counter += 1
    # 1. get all topics
    current_topics = embedding_df["topic"].unique().tolist()  # [0,1,2,3,4,5,6]
    # print(current_topics)
    topic_pair_combinations = list(
        itertools.combinations(current_topics, 2)
    )  # [(0, 1), (0, 2), (3, 4), (3, 5)]
    embedding_df_new, flatten_df_new = merge_clusters(
        topic_pair_combinations, th, embedding_df, flatten_df
    )  # [[0, 1, 2], [3, 4, 5]] # type: ignore
    if embedding_df_new is not None:
        return recursive_merge_clusters(embedding_df_new, flatten_df_new, th)
    else:
        # print(print(embedding_df["topic"].unique().tolist()))
        return embedding_df, flatten_df


# embedding_df, flatten_df = recursive_merge_clusters(embedding_df, flatten_df)


def update_topic(df, flatten_df):
    # make a copy of df to avoid modifying the original dataframe
    new_df = df.copy()
    # loop through rows of flatten_df and update new_df
    for idx, row in flatten_df.iterrows():
        text = row["text"]
        topic = row["topic"]

        mask1 = new_df["neighbor"] == text
        mask2 = new_df["node"] == text
        new_df.loc[mask1, "topic"] = topic
        new_df.loc[mask2, "topic"] = topic

    return new_df


# def remove_outliers(embedding_df, threshold):
#     embeddings = embedding_df.to_numpy()
#     distance_matrix = cdist(embeddings, embeddings, "cosine")

#     similarity_matrix = 1 - distance_matrix
#     similarity_matrix[np.isnan(similarity_matrix)] = 0

#     outlier_indices = []
#     for i in range(len(embeddings)):
#         similarity_scores = similarity_matrix[i, :]
#         num_similar_embeddings = np.sum(similarity_scores >= threshold)

#         if num_similar_embeddings <= 1:  # Exclude the embedding itself
#             outlier_indices.append(i)

#     filtered_embeddings = np.delete(embeddings, outlier_indices, axis=0)
#     filtered_df = pd.DataFrame(filtered_embeddings)

#     return filtered_df


def get_most_outdegree_node(graph):
    # Calculate out-degree for each node in the graph
    outdegrees = graph.out_degree()

    # Find the node with the maximum out-degree
    most_outdegree_node = max(outdegrees, key=lambda x: x[1])

    return most_outdegree_node[0]


def get_weakly_connected_subgraphs(df):
    # Create an empty directed graph
    graph = nx.DiGraph()

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        topic = row["topic"]
        node = row["node"]
        neighbor = row["neighbor"]

        # Add nodes to the graph
        graph.add_node(node, topic=topic)
        graph.add_node(neighbor, topic=topic)

        # Add edges to the graph
        graph.add_edge(node, neighbor, topic=topic)

    # Find weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(graph))

    # Create a list to store subgraphs
    subgraphs = []

    # Iterate over each weakly connected component
    for component in weakly_connected_components:
        # Create a subgraph for the component
        subgraph = graph.subgraph(component)
        subgraphs.append(subgraph)

    return subgraphs


def process_topics(df):
    for topic in df["topic"].unique().tolist():
        topic_df = df[df["topic"] == topic]
        subgraphs = get_weakly_connected_subgraphs(topic_df)

        # find nodes with most outdegree from each sub graphs
        most_outdegree_node_str_list = []
        for i, subgraph in enumerate(subgraphs):
            most_outdegree_node_str = get_most_outdegree_node(subgraph)
            most_outdegree_node_str_list.append(most_outdegree_node_str)

        # Create a bidirectional link between them
        for node in most_outdegree_node_str_list:
            for neighbor in most_outdegree_node_str_list:
                if node != neighbor:
                    # node_info = df[df["node"] == node].iloc[0]
                    # neighbor_info = df[df["node"] == neighbor].iloc[0]
                    link = {
                        "topic": topic,
                        "node": node,
                        "neighbor": neighbor,
                        # "node_year": node_info["node_year"],
                        # "neighbor_year": neighbor_info["neighbor_year"],
                        # "node_paper_id": node_info["node_paper_id"],
                        # "neighbor_paper_id": neighbor_info["neighbor_paper_id"],
                    }
                    df = pd.concat([df, pd.DataFrame([link])], ignore_index=True)
    return df


# def _print_topics(flatten_df):
#     # text groupby
#     Domains_text_list = (
#         flatten_df.groupby("topic")["text"].apply(list).reset_index()["text"].tolist()
#     )
#     # list of list to list of string
#     Domains_text_list = [" ".join(each) for each in Domains_text_list]
#     for text in Domains_text_list:
#         print(text)
#         print()


def _print_topics(df):
    topic_keyphrases = (
        df.groupby("topic")["text"].apply(list).reset_index().values.tolist()
    )

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns for each topic
    for topic, keyphrases in topic_keyphrases:
        table.add_column(str(topic))

    # Get the maximum number of keyphrases for any topic
    max_keyphrases = max(len(keyphrases) for _, keyphrases in topic_keyphrases)

    # Add rows for keyphrases for each topic
    for i in range(max_keyphrases):
        row = []
        for _, keyphrases in topic_keyphrases:
            if i < len(keyphrases):
                row.append(keyphrases[i])
            else:
                row.append("")  # Fill empty cells with empty string
        table.add_row(*row)

    console.print(table)
    console.print()


# Rest of the code...


def flatten_graph_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df: DataFrame(cols=[topic,node,neighbor])
    Output df :-> DataFrame(cols=[topic,text])

    """
    new_doc: List[Dict[str, str]] = []
    for topic in df["topic"].unique().tolist():
        topic_df = df[df["topic"] == topic]
        nodes = topic_df["node"].unique().tolist()
        neighbors = topic_df["neighbor"].unique().tolist()
        doc = list(set(neighbors + nodes))
        data = [{"topic": topic, "text": v} for v in doc]
        new_doc.extend(data)
    flatten_df = pd.DataFrame(new_doc)
    return flatten_df


def flatten_df_to_dict(flatten_df: pd.DataFrame) -> Dict[int, List[str]]:
    result_dict: Dict[int, List[str]] = {}
    for topic in flatten_df["topic"].unique().tolist():
        topic_texts = flatten_df[flatten_df["topic"] == topic]["text"].tolist()
        result_dict[topic] = topic_texts
    return result_dict


def topic_dict_to_list(topic_dict, max_topics=None):
    if not max_topics:
        return list(topic_dict.values())
    else:
        return list(topic_dict.values())[:max_topics]


def get_merged_topics(
    latent_graph_df: pd.DataFrame,
    embeddingModel: SentenceTransformer,
    threshold=0.5,
    pretty_print=True,
):
    """
    Corresponding Columns:
    latent_graph_df,updated_graph_df :->  DataFrame(cols=[topic,node,neighbor])
    flatten_df,merged_df :-> DataFrame(cols=[topic,text])
    """
    flatten_df = flatten_graph_df(latent_graph_df)
    phrase_embs = embeddingModel.encode(flatten_df["text"].tolist())
    embedding_df = pd.DataFrame(np.array(phrase_embs.tolist()))  # type: ignore
    embedding_df["topic"] = flatten_df["topic"]
    _, merged_df = recursive_merge_clusters(embedding_df, flatten_df, threshold)
    # if pretty_print:
    #     print(merged_df.shape)
    #     _print_topics(merged_df)
    #     # print(updated_graph_df.shape)  # type: ignore

    # Also update original `df`
    updated_graph_df = update_topic(latent_graph_df, merged_df)
    # print(updated_graph_df.shape)
    # make connection between subgraphs within same topics
    final_graph_df = process_topics(updated_graph_df)
    final_flatten_df = flatten_graph_df(final_graph_df)

    _topics: Dict[int, List[str]] = {}

    _topics = flatten_df_to_dict(final_flatten_df)

    return final_graph_df, final_flatten_df, topic_dict_to_list(_topics)


# Get top keyphrases using cosine distance
def get_top_cdist_kps(
    kps: List[str], embeddingModel: SentenceTransformer, max_keyphrases_per_topic=10
):
    embeddings = embeddingModel.encode(kps)
    distance_matrix = cdist(embeddings, embeddings, "cosine")

    similar_indices = np.argsort(np.sum(distance_matrix, axis=0))[
        :max_keyphrases_per_topic
    ]
    similar_kps = [kps[i] for i in similar_indices]

    # filtered_embeddings = np.delete(embeddings, similar_indices, axis=0)
    # filtered_df = pd.DataFrame(filtered_embeddings)
    # return similar_kps, filtered_df

    return similar_kps


class KeyphrasesSelectionStrategy(Enum):
    IMP_NODE = "outdegree"
    CDIST = "cdist"
    RNDOM = "random"


def get_topics_with_pruned_kps(
    flatten_df,
    embeddingModel: SentenceTransformer,
    max_keyphrases_per_topic=10,
    keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST,
    max_topics=None,
):
    # Initialize the topics dictionary
    _topics = {}
    # Group the text column by the topic column
    grouped = flatten_df.groupby("topic")["text"]
    # Iterate over each group
    for topic, group in grouped:
        # Extract the list of texts for the current topic
        kps = group.tolist()
        # Add the topic and its corresponding texts to the topics dictionary
        if keyphrases_selection_strategy == KeyphrasesSelectionStrategy.IMP_NODE:
            # Handle "first" strategy
            _topics[topic] = kps[:max_keyphrases_per_topic]
        elif keyphrases_selection_strategy == KeyphrasesSelectionStrategy.CDIST:
            # Handle "cdist" strategy
            _topics[topic] = get_top_cdist_kps(
                kps, embeddingModel, max_keyphrases_per_topic
            )
        elif keyphrases_selection_strategy == KeyphrasesSelectionStrategy.RNDOM:
            # Handle "random" strategy
            _topics[topic] = random.sample(kps, min(max_keyphrases_per_topic, len(kps)))
        else:
            # Handle other cases or raise an exception for invalid strategy
            raise ValueError("Invalid keyphrases_selection_strategy")

    # pprint(_topics)

    return topic_dict_to_list(_topics, max_topics=max_topics)
