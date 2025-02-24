from GraphTopic.helpers.text_preprocess import extract_noun_phrases, _preprocess_text
from GraphTopic.helpers.config import embeddingModel
from GraphTopic import GraphTopic

from others.models import (
    preprocess_gensim,
    get_lda_topics,
    get_lsi_topics,
    get_hdp_topics,
    get_nmf_topics,
    get_bertopic_domains_with_kps,
)
from scores import (
    calculate_coherence_scores,
    get_semantic_score,
    calculate_topic_diversity,
)

from pprint import pprint
from data.prepare import (
    get_sample_data_chunks,
    get_result_data_20_newsgroups,
    get_result_data_trump,
    get_result_data_bbc,
)

import warnings
import pandas as pd

warnings.filterwarnings("ignore")


max_topics = 10
max_keyphrases_per_topic = 10
TopM = 500
TopN = 10
overlapping_threshold = 0.5
text_chunks = get_sample_data_chunks()
# text_chunks = get_result_data_20_newsgroups()
# text_chunks = get_result_data_trump()
# text_chunks = get_result_data_bbc()


if __name__ == "__main__":
    all_results = []

    for text_chunk in text_chunks:
        docs = [extract_noun_phrases(doc) for doc in text_chunk[:30]]
        docs = _preprocess_text(docs, deep_clean=True)
        corpus, dictionary = preprocess_gensim(docs)

        # Bertopic
        try:
            bert_topics = get_bertopic_domains_with_kps(
                docs, words_per_topic=max_keyphrases_per_topic
            )
            topic_count = min(len(bert_topics), max_topics)
            bert_topics = bert_topics[:topic_count]
            bert_coherence = calculate_coherence_scores(
                bert_topics, docs, topn=max_keyphrases_per_topic
            )
            bert_semantic_coherence = get_semantic_score(embeddingModel, bert_topics)
            bert_diversity_score = calculate_topic_diversity(bert_topics)

            bert_results = {
                "Method": "Bertopic",
                "c_npmi": bert_coherence.get("c_npmi", None),
                "c_uci": bert_coherence.get("c_uci", None),
                "c_v": bert_coherence.get("c_v", None),
                "u_mass": bert_coherence.get("u_mass", None),
                "semantic_coherence": bert_semantic_coherence.get("coherence", None),
                "inter_distances_avg": bert_semantic_coherence.get(
                    "inter_distances_avg", None
                ),
                "intra_similarity_avg": bert_semantic_coherence.get(
                    "intra_similarity_avg", None
                ),
                "diversity_score": bert_diversity_score.get("topic_diversity", None),
            }
            all_results.append(bert_results)

        except Exception as e:
            print("Bertopic:error")
            print(e)

        # LDA, LSI, HDP, NMF
        lda_topics = get_lda_topics(
            corpus,
            dictionary,
            num_topics=max_topics,
            num_words=max_keyphrases_per_topic,
        )
        lsi_topics = get_lsi_topics(
            corpus,
            dictionary,
            num_topics=max_topics,
            num_words=max_keyphrases_per_topic,
        )
        hdp_topics = get_hdp_topics(
            corpus,
            dictionary,
            num_topics=max_topics,
            num_words=max_keyphrases_per_topic,
        )
        nmf_topics = get_nmf_topics(
            corpus,
            dictionary,
            num_topics=max_topics,
            num_words=max_keyphrases_per_topic,
        )

        # Calculate scores for LDA
        lda_coherence = calculate_coherence_scores(
            lda_topics, docs, topn=max_keyphrases_per_topic
        )
        lda_semantic_coherence = get_semantic_score(embeddingModel, lda_topics)
        lda_diversity_score = calculate_topic_diversity(lda_topics)
        lda_results = {
            "Method": "LDA",
            "c_npmi": lda_coherence.get("c_npmi", None),
            "c_uci": lda_coherence.get("c_uci", None),
            "c_v": lda_coherence.get("c_v", None),
            "u_mass": lda_coherence.get("u_mass", None),
            "semantic_coherence": lda_semantic_coherence.get("coherence", None),
            "inter_distances_avg": lda_semantic_coherence.get(
                "inter_distances_avg", None
            ),
            "intra_similarity_avg": lda_semantic_coherence.get(
                "intra_similarity_avg", None
            ),
            "diversity_score": lda_diversity_score.get("topic_diversity", None),
        }
        all_results.append(lda_results)

        # Calculate scores for LSI
        lsi_coherence = calculate_coherence_scores(
            lsi_topics, docs, topn=max_keyphrases_per_topic
        )
        lsi_semantic_coherence = get_semantic_score(embeddingModel, lsi_topics)
        lsi_diversity_score = calculate_topic_diversity(lsi_topics)
        lsi_results = {
            "Method": "LSI",
            "c_npmi": lsi_coherence.get("c_npmi", None),
            "c_uci": lsi_coherence.get("c_uci", None),
            "c_v": lsi_coherence.get("c_v", None),
            "u_mass": lsi_coherence.get("u_mass", None),
            "semantic_coherence": lsi_semantic_coherence.get("coherence", None),
            "inter_distances_avg": lsi_semantic_coherence.get(
                "inter_distances_avg", None
            ),
            "intra_similarity_avg": lsi_semantic_coherence.get(
                "intra_similarity_avg", None
            ),
            "diversity_score": lsi_diversity_score.get("topic_diversity", None),
        }
        all_results.append(lsi_results)

        # Calculate scores for HDP
        hdp_coherence = calculate_coherence_scores(
            hdp_topics, docs, topn=max_keyphrases_per_topic
        )
        hdp_semantic_coherence = get_semantic_score(embeddingModel, hdp_topics)
        hdp_diversity_score = calculate_topic_diversity(hdp_topics)
        hdp_results = {
            "Method": "HDP",
            "c_npmi": hdp_coherence.get("c_npmi", None),
            "c_uci": hdp_coherence.get("c_uci", None),
            "c_v": hdp_coherence.get("c_v", None),
            "u_mass": hdp_coherence.get("u_mass", None),
            "semantic_coherence": hdp_semantic_coherence.get("coherence", None),
            "inter_distances_avg": hdp_semantic_coherence.get(
                "inter_distances_avg", None
            ),
            "intra_similarity_avg": hdp_semantic_coherence.get(
                "intra_similarity_avg", None
            ),
            "diversity_score": hdp_diversity_score.get("topic_diversity", None),
        }
        all_results.append(hdp_results)

        # Calculate scores for NMF
        nmf_coherence = calculate_coherence_scores(
            nmf_topics, docs, topn=max_keyphrases_per_topic
        )
        nmf_semantic_coherence = get_semantic_score(embeddingModel, nmf_topics)
        nmf_diversity_score = calculate_topic_diversity(nmf_topics)
        nmf_results = {
            "Method": "NMF",
            "c_npmi": nmf_coherence.get("c_npmi", None),
            "c_uci": nmf_coherence.get("c_uci", None),
            "c_v": nmf_coherence.get("c_v", None),
            "u_mass": nmf_coherence.get("u_mass", None),
            "semantic_coherence": nmf_semantic_coherence.get("coherence", None),
            "inter_distances_avg": nmf_semantic_coherence.get(
                "inter_distances_avg", None
            ),
            "intra_similarity_avg": nmf_semantic_coherence.get(
                "intra_similarity_avg", None
            ),
            "diversity_score": nmf_diversity_score.get("topic_diversity", None),
        }
        all_results.append(nmf_results)

        # GraphTopic
        try:
            model = GraphTopic(
                docs,
                max_topics=max_topics,
                TopM=TopM,
                TopN=TopN,
                overlapping_threshold=overlapping_threshold,
            )
            topics = model.get_topics()
            m_coherence = calculate_coherence_scores(
                topics, docs, topn=max_keyphrases_per_topic
            )
            m_semantic_coherence = get_semantic_score(embeddingModel, topics)
            m_diversity_score = calculate_topic_diversity(topics)

            gt_results = {
                "Method": "GraphTopic",
                "c_npmi": m_coherence.get("c_npmi", None),
                "c_uci": m_coherence.get("c_uci", None),
                "c_v": m_coherence.get("c_v", None),
                "u_mass": m_coherence.get("u_mass", None),
                "semantic_coherence": m_semantic_coherence.get("coherence", None),
                "inter_distances_avg": m_semantic_coherence.get(
                    "inter_distances_avg", None
                ),
                "intra_similarity_avg": m_semantic_coherence.get(
                    "intra_similarity_avg", None
                ),
                "diversity_score": m_diversity_score.get("topic_diversity", None),
            }
            all_results.append(gt_results)

        except Exception as e:
            print("GraphTopic:error")
            print(e)

    df_results = pd.DataFrame(all_results)
    average_results = df_results.groupby("Method").mean()
    print("Average Results Across All Text Chunks:")
    print(average_results)
    # save
    df_results.to_csv("results.csv")
    average_results.to_csv("average_results.csv")
