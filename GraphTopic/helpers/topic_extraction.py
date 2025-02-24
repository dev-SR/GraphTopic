import os

from pprint import pprint
import pandas as pd
import numpy as np
from netgraph import Graph
import warnings

warnings.filterwarnings("ignore")


from pprint import pprint
from copy import copy
from typing import TypeVar, List, Dict, Union, Literal
from dataclasses import dataclass
from collections import deque
from scipy.spatial.distance import cdist

import networkx as nx
import math

# Create a graph object
import matplotlib.pyplot as plt


@dataclass
class KeyPhrase:
    paper_id: str
    keyphrase: str
    year: int = 0
    score: float = 0

    def __hash__(self):
        return hash(self.keyphrase)

    def __eq__(self, other):
        if isinstance(other, KeyPhrase):
            return self.keyphrase == other.keyphrase
        return False


NodeType = TypeVar("NodeType", bound="Node")
InfoType = Dict[Literal["keyphrase", "topic_index"], Union[KeyPhrase, int]]
AdjacencyList = Dict[NodeType, List[NodeType]]


class Node:
    def __init__(self, data: KeyPhrase):
        self._paper_id = data.paper_id
        self._keyphrase = data.keyphrase
        self._year = data.year

    def __str__(self):
        return f"Node({self._keyphrase})"

    def __repr__(self):
        return str(self)

    @property
    def id(self):
        return self._paper_id

    @property
    def keyphrase(self):
        return copy(self._keyphrase)

    @property
    def year(self):
        return self._year

    def __eq__(self, other: NodeType):
        return True if self._keyphrase == other.keyphrase else False

    def __hash__(self):
        return hash(self._keyphrase)


class TopicManager:
    def __init__(
        self,
    ):
        self._topics: List[AdjacencyList] = []
        self._csv_reports = pd.DataFrame(
            columns=["topic", "node", "neighbor", "weight"]
        )
        self._all_kps_found_history = []

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self._topics)

    def add_new_topic_from_keyphrases(self, kps: list[KeyPhrase]) -> None:
        adj_list = {}
        for item in kps:
            node = Node(item)
            adj_list[node] = []
            for neighbor in kps:
                if neighbor.keyphrase != node.keyphrase:
                    neighbor_node = Node(neighbor)
                    adj_list[node].append((neighbor_node, 1))
        self._topics.append(adj_list)

    def print_details(self):
        for i, topic in enumerate(self._topics):
            print(f"Domain[{i}]")
            for key in topic.keys():
                print(f'   {key}:{"-" * (25 - len(str(key)))}->{topic[key]}')
        print()

    def __get_topic_indices_by_keyphrase(self, kp: KeyPhrase) -> list[tuple[int]]:
        index_list = []
        for i, adj_list in enumerate(self._topics):
            for node in adj_list.keys():
                if node.keyphrase == kp.keyphrase:
                    index_list.append(i)
        return index_list

    def __make_connectivity(self, nodes: List[NodeType], graph: AdjacencyList):
        # Create a set of all the nodes in the adjacency list
        all_nodes = list(graph.keys())
        get_common_nodes = list(set(all_nodes) & set(nodes))
        # print(get_common_nodes)
        for node in get_common_nodes:
            for node_to_add in nodes:
                if node_to_add != node:
                    adj_nodes = graph[node]
                    if node_to_add not in self.__get_only_nodes(adj_nodes):
                        adj_nodes.append((node_to_add, 1))

    def __merge_graphs(self, graph1, graph2, merged_graph):
        # Iterate over the adjacency lists of both graphs
        for graph in [graph1, graph2]:
            for vertex, edges in graph.items():
                # Check if the vertex already exists in the new adjacency list
                if vertex in merged_graph:
                    # Add the unique edges from the entry to the list of edges for the vertex
                    merged_graph[vertex] += list(set(edges) - set(merged_graph[vertex]))
                else:
                    # Add a new entry to the list with the vertex and its edges
                    merged_graph[vertex] = edges
        return merged_graph

    # def get_node_from_topic(self, kp: KeyPhrase):
    #     for adj_list in self._topics:
    #         for node in adj_list.keys():
    #             if node.keyphrase == kp.keyphrase:
    #                 return node

    # def get_topic_by_node(self, node: NodeType):
    #     for i, adj_list in enumerate(self._topics):
    #         for n in adj_list.keys():
    #             if n == node:
    #                 return i

    # def get_keyphrase_obj_from_list(self, kps_list):
    #     kps_obj_list = []
    #     for kp_string in kps_list:
    #         kp_job = KeyPhrase("", kp_string)
    #         kps_obj_list.append(kp_job)
    #     return kps_obj_list

    def __get_only_nodes(self, nodes_weight_tuple_list) -> List[NodeType]:
        return [t[0] for t in nodes_weight_tuple_list]

    def merge_subtopics(self, kps: list[KeyPhrase]):
        # find topics that carry the passed keyphrases
        found_list: List[InfoType] = []
        for kp in kps:
            indices = self.__get_topic_indices_by_keyphrase(kp)
            if indices:
                for index in indices:
                    info: InfoType = {
                        "keyphrase": kp,
                        "topic_index": index,
                    }
                    found_list.append(info)
        # pprint(found_list)
        if not found_list:
            return

        # get unique topics
        unique_topic = set()
        unique_kps = set()
        for item in found_list:
            unique_topic.add(item["topic_index"])
            unique_kps.add(item["keyphrase"])
        unique_topic_list = list(unique_topic)
        unique_kps = list(unique_kps)
        # print(unique_kp)

        if len(unique_topic_list) < 2:
            return
        # unique_topic_list = sorted(unique_topic_list)
        base_topic_index = unique_topic_list[0]
        to_merge_indices = unique_topic_list[1:]

        if len(to_merge_indices) >= 1:
            q = deque(to_merge_indices)
            while q:
                merged_graph = {}
                topic_to_merge_index = q.popleft()
                graph1 = self._topics[base_topic_index]
                graph2 = self._topics[topic_to_merge_index]
                self._topics[base_topic_index] = self.__merge_graphs(
                    graph1, graph2, merged_graph
                )

        self.__make_connectivity(
            [Node(kp) for kp in unique_kps], self._topics[base_topic_index]
        )
        # print(self._topics[base_topic_index])

        # remove from self._topics
        if to_merge_indices:
            for index in sorted(
                to_merge_indices, reverse=True
            ):  # reversed because list shrinks when popped
                try:
                    self._topics.pop(index)
                except Exception as e:
                    print(unique_topic)
                    print(index)
                    print(e)

    def clear_topic_with_node_count_n(self, n):
        self._topics = [d for d in self._topics if len(d) > n]
        self._topics = sorted(self._topics, key=lambda d: len(d), reverse=True)

    def get_topics(self, kps_per_topic=None):
        data = {}
        for i, topic in enumerate(self._topics):
            all_kps = set()
            for node in topic.keys():
                all_kps.add(node.keyphrase)
                for neighbor, weight in topic[node]:
                    all_kps.add(neighbor.keyphrase)
            data[i] = list(all_kps)[:kps_per_topic]
        return data

    def generate_csv(self, path):
        for i, topic in enumerate(self._topics):
            data = []
            for node in topic.keys():
                for neighbor, weight in topic[node]:
                    node_info = {}
                    node_info["topic"] = i
                    node_info["node"] = node.keyphrase
                    node_info["neighbor"] = neighbor.keyphrase
                    node_info["weight"] = weight
                    node_info["node_year"] = node.year
                    node_info["neighbor_year"] = neighbor.year
                    node_info["node_paper_id"] = node.id
                    node_info["neighbor_paper_id"] = neighbor.id
                    data.append(node_info)
            topic_df = pd.DataFrame(data)
            # concat
            self._csv_reports = pd.concat(
                [self._csv_reports, topic_df], ignore_index=True
            )
            #
            self.latent_df = pd.concat([self.latent_df, topic_df], ignore_index=True)

        self._csv_reports.to_csv(path, index=False)

    def generate_df(self):
        latent_df = pd.DataFrame(columns=["topic", "node", "neighbor"])
        for i, topic in enumerate(self._topics):
            data = []
            for node in topic.keys():
                for neighbor, weight in topic[node]:
                    node_info = {}
                    node_info["topic"] = i
                    node_info["node"] = node.keyphrase
                    node_info["neighbor"] = neighbor.keyphrase
                    data.append(node_info)
            topic_df = pd.DataFrame(data)
            latent_df = pd.concat([latent_df, topic_df], ignore_index=True)
        return latent_df

    @property
    def topics(self):
        return copy(self._topics)

    def draw_topics(self, show=True, save=False, save_path="", drawCluster=False):
        n = len(self._topics)
        n = n + 1 if n == 1 else n
        if drawCluster:
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(5 * n, 5 * n))
        else:
            fig, axes = plt.subplots(
                nrows=n, ncols=1, figsize=(20, (20 + (15 * n)))
            )  # (20,  (20 + (10*n))))
        ax = axes.flatten()
        subgraphs = []
        for topic in self._topics:
            G = nx.Graph()
            nodes = set()
            for node in topic.keys():
                G.add_node(node.keyphrase)
                nodes.add(node.keyphrase)
                for adj_node, weight in topic[node]:
                    G.add_edge(node.keyphrase, adj_node.keyphrase)
                    nodes.add(adj_node.keyphrase)

            subgraph = nx.subgraph(G, list(nodes))
            subgraphs.append(subgraph)

        for i in range(len(subgraphs)):
            G = subgraphs[i]
            pos = nx.kamada_kawai_layout(
                G
            )  # kamada_kawai_layout | spring_layout(G,k=0.03)
            # Specify node positions using pos argument
            nx.draw(
                G,
                pos=pos,
                ax=ax[i],
                node_size=30,
                edge_color="gray",
                node_color="skyblue",
                width=0.5,
            )
            # Specify offset parameter to push labels up 2 pixels
            # label_pos = {node: (x, y + .01) for node, (x, y) in pos.items()}
            nx.draw_networkx_labels(
                G, pos=pos, ax=ax[i], font_size=6, font_color="black"
            )
            # Set the title for each subplot with a larger font size and red color
            if drawCluster:
                ax[i].set_title(f"Cluster {i}", fontsize=16, color="red")
            else:
                ax[i].set_title(f"Domain {i}", fontsize=16, color="red")

        if save_path != "":
            plt.savefig(save_path, format="pdf", dpi=300, transparent=True)

        if show:
            plt.show()
        else:
            plt.close()
