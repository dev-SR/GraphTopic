from datasets import load_dataset
from GraphTopic.helpers.text_preprocess import split_into_chunks

import pandas as pd
from pathlib import Path


def get_result_data_20_newsgroups():
    # SetFit/20_newsgroups
    # SetFit/bbc-news
    # SetFit/imdb
    dataset_name = "SetFit/20_newsgroups"
    dataset = load_dataset(dataset_name)

    # Extract training and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Extract text data from the training split
    train_docs = [data["text"] for data in train_data]

    # Extract text data from the test split
    test_docs = [data["text"] for data in test_data]

    # Merge test split into docs
    raw_docs = train_docs + test_docs

    train_len = len(train_docs)
    test_len = len(test_docs)
    total_docs_len = train_len + test_len
    # print("Number of documents in the training set:", train_len)
    # print("Number of documents in the test set:", test_len)
    print("Total number of documents:", total_docs_len)

    chunk_size = 3768
    overlap_ratio = 0
    v = raw_docs[:-6]
    text_chunks = split_into_chunks(
        v, chunk_size=chunk_size, overlap_ratio=overlap_ratio
    )

    return text_chunks


def get_result_data_bbc():
    dataset_name = "SetFit/bbc-news"
    dataset = load_dataset(dataset_name)
    # Extract training and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Extract text data from the training split
    train_docs = [data["text"] for data in train_data]

    # Extract text data from the test split
    test_docs = [data["text"] for data in test_data]

    # Merge test split into docs
    raw_docs = train_docs + test_docs

    train_len = len(train_docs)
    test_len = len(test_docs)
    total_docs_len = train_len + test_len
    # print("Number of documents in the training set:", train_len)
    # print("Number of documents in the test set:", test_len)
    print("Total number of documents:", total_docs_len)

    chunk_size = 2225
    overlap_ratio = 0
    text_chunks = split_into_chunks(
        raw_docs, chunk_size=chunk_size, overlap_ratio=overlap_ratio
    )
    return text_chunks


def get_result_data_trump():
    df = pd.read_csv("data/tweets_01-08-2021.csv")
    raw_docs = df["text"].tolist()[:-1]
    chunk_size = 11314
    overlap_ratio = 0
    text_chunks = split_into_chunks(
        raw_docs, chunk_size=chunk_size, overlap_ratio=overlap_ratio
    )
    return text_chunks


def read_documents_from_dir(directory: str):
    docs_dir = Path(directory)
    docs = []
    for doc_path in docs_dir.glob("*.txt"):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(text)
    return docs


def get_sample_data():
    directory = "data/docs"
    docs = read_documents_from_dir(directory)
    return docs


def get_sample_data_chunks():
    directory = "data/docs"
    docs = read_documents_from_dir(directory)
    return [docs]
