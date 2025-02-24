# https://github.com/MaartenGr/BERTopic/issues/90
# https://radimrehurek.com/gensim/models/coherencemodel.html
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from typing import List, Dict
import numpy as np


def calculate_coherence_scores(
    topics: List[List[str]], docs: List[str], topn=20
) -> Dict[str, float]:
    # https://stackoverflow.com/questions/38115367/scikit-learn-dont-separate-hyphenated-words-while-tokenization
    token_pattern = "(?u)\\b[\\w-]+\\b"
    vectorizer_model = CountVectorizer(
        stop_words="english", ngram_range=(2, 3), token_pattern=token_pattern
    )
    vectorizer_model.fit(docs)
    analyzer = vectorizer_model.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]

    topic_words = []
    for topic in topics:
        words = [word for word in topic if word in dictionary.token2id]
        topic_words.append(words)
    topic_words = [words for words in topic_words if len(words) > 0]

    coherence_scores = {}
    for measure in ["c_v", "u_mass", "c_uci", "c_npmi"]:
        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            corpus=corpus,
            dictionary=dictionary,
            coherence=measure,
            topn=topn,
        )
        # coherence_score = coherence_model.get_coherence()
        s = coherence_model.get_coherence_per_topic()
        coherence_scores[measure] = np.mean(s)
        # nan_values = np.isnan(s)
        # nan_values = np.sum(nan_values)
        # https://github.com/piskvorky/gensim/issues/3040
        # s = np.nan_to_num(s, nan=np.nanmean(s))
        # print(s)
        # Count NaN values
        # coherence_scores[f"{measure}_nan"] = nan_values

    # print(coherence_scores)
    return coherence_scores


def get_avg_intra_score(model, topics, debug=False):
    def calculate_cosine_similarity(texts):
        embeddings = model.encode(texts, convert_to_tensor=True)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    avg_intra_similarities = []

    for topic in topics:
        similarity_matrix = calculate_cosine_similarity(topic)
        intra_similarity = np.mean(similarity_matrix)
        avg_intra_similarities.append(intra_similarity)

    overall_avg_intra_similarity = np.mean(avg_intra_similarities)
    if debug:
        print("Average Intra-Similarities for Each Topic:")
        for i, topic in enumerate(topics):
            print(f"Topic {i + 1}: {avg_intra_similarities[i]:.4f}")

        print(f"Overall Average Intra-Similarity: {overall_avg_intra_similarity:.4f}\n")

    return overall_avg_intra_similarity


def get_avg_inter_score(model, topics, debug=False):
    if len(topics) < 2:
        return 0.0

    # Function to calculate cosine similarity
    def calculate_inter_distances(texts1, texts2):
        embeddings1 = model.encode(
            texts1, convert_to_tensor=True
        )  # Shape:[Topic_Size1,Embedding_Dim]
        embeddings2 = model.encode(
            texts2, convert_to_tensor=True
        )  # Shape:[Topic_Size2,Embedding_Dim]
        similarity_matrix = cosine_similarity(
            embeddings1, embeddings2
        )  # Shape: [Topic_Size1,Topic_Size2]
        inter_distances = 1 - similarity_matrix  # Calculate inter-distances
        return inter_distances

    inter_distances = []

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            inter_distance_matrix = calculate_inter_distances(topics[i], topics[j])
            inter_distance = np.mean(inter_distance_matrix)
            inter_distances.append(inter_distance)

    overall_avg_inter_distance = np.mean(inter_distances)

    # print(inter_distances)
    if debug:
        print("Inter-Distances between Topics:")
        pair_count = 0
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                print(f"Topics {i + 1} and {j + 1}: {inter_distances[pair_count]:.4f}")
                pair_count += 1

        print(f"Overall Average Inter-Distance: {overall_avg_inter_distance:.4f}\n")

    return overall_avg_inter_distance


def get_semantic_score(embeddingModel, topics_list, debug=False):
    intra_avg = get_avg_intra_score(embeddingModel, topics_list, debug)
    inter_avg = get_avg_inter_score(embeddingModel, topics_list, debug)
    coherence = np.array([intra_avg, inter_avg]).mean()

    return {
        "intra_similarity_avg": intra_avg,  # Higher Better
        "inter_distances_avg": inter_avg,  # Higher Better
        "coherence": coherence,  # Coherence
    }


def calculate_topic_diversity(topics_list) -> float:
    total_unique_words = set()
    total_words = []
    for topic_words in topics_list:
        total_words.extend(topic_words)
        total_unique_words.update(topic_words)

    total_unique_words_should_be = len(total_words)
    total_unique_words_found = len(total_unique_words)
    diversity_score = total_unique_words_found / total_unique_words_should_be
    return {"topic_diversity": diversity_score}
