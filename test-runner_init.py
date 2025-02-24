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

warnings.filterwarnings("ignore")


max_topics = 10
max_keyphrases_per_topic = 10
TopM = 500
TopN = 10
overlapping_threshold = 0.5

if __name__ == "__main__":
    text_chunks = get_sample_data_chunks()
    # text_chunks = get_result_data_20_newsgroups()
    # text_chunks = get_result_data_trump()
    # text_chunks = get_result_data_bbc()

    for text_chunk in text_chunks:
        docs = [extract_noun_phrases(doc) for doc in text_chunk[:30]]
        docs = _preprocess_text(docs, deep_clean=True)
        corpus, dictionary = preprocess_gensim(docs)

        try:
            bert_topics = get_bertopic_domains_with_kps(
                docs, words_per_topic=max_keyphrases_per_topic
            )
            topic_count = min(len(bert_topics), max_topics)
            bert_topics = bert_topics[:topic_count]
            print("Calculating statistical coherence scores")
            bert_coherence = calculate_coherence_scores(
                bert_topics, docs, topn=max_keyphrases_per_topic
            )
            print("Calculating semantics similarity scores")
            bert_semantic_coherence = get_semantic_score(embeddingModel, bert_topics)
            print("Calculating topic diversity scores")
            bert_diversity_score = calculate_topic_diversity(bert_topics)

            print("Bertopic Coherence Scores:")
            pprint(bert_coherence)
            print("Bertopic Semantic Scores:")
            pprint(bert_semantic_coherence)
            print("Bertopic Diversity Score:")
            pprint(bert_diversity_score)

        except Exception as e:
            print("Bertopic:error")
            print(e)

        print("Calculating LDA, LSI, HDP, NMF topics...")
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
        # Calculate statistical coherence scores for each method
        print("Calculating statistical coherence scores")
        lda_coherence = calculate_coherence_scores(
            lda_topics, docs, topn=max_keyphrases_per_topic
        )
        lsi_coherence = calculate_coherence_scores(
            lsi_topics, docs, topn=max_keyphrases_per_topic
        )
        hdp_coherence = calculate_coherence_scores(
            hdp_topics, docs, topn=max_keyphrases_per_topic
        )
        nmf_coherence = calculate_coherence_scores(
            nmf_topics, docs, topn=max_keyphrases_per_topic
        )
        # Calculate semantics similarity scores for each method
        print("Calculating semantics similarity scores")
        lda_semantic_coherence = get_semantic_score(embeddingModel, lda_topics)
        lsi_semantic_coherence = get_semantic_score(embeddingModel, lsi_topics)
        hdp_semantic_coherence = get_semantic_score(embeddingModel, hdp_topics)
        nmf_semantic_coherence = get_semantic_score(embeddingModel, nmf_topics)
        # Calculate topic diversity scores
        print("Calculating topic diversity scores")
        lda_diversity_score = calculate_topic_diversity(lda_topics)
        lsi_diversity_score = calculate_topic_diversity(lsi_topics)
        hdp_diversity_score = calculate_topic_diversity(hdp_topics)
        nmf_diversity_score = calculate_topic_diversity(nmf_topics)

        print("LDA Coherence Scores:")
        pprint(lda_coherence)
        print("LDA Semantic Scores:")
        pprint(lda_semantic_coherence)
        print("LDA Diversity Score:")
        pprint(lda_diversity_score)

        print("LSI Coherence Scores:")
        pprint(lsi_coherence)
        print("LSI Semantic Scores:")
        pprint(lsi_semantic_coherence)
        print("LSI Diversity Score:")
        pprint(lsi_diversity_score)

        print("HDP Coherence Scores:")
        pprint(hdp_coherence)
        print("HDP Semantic Scores:")
        pprint(hdp_semantic_coherence)
        print("HDP Diversity Score:")
        pprint(hdp_diversity_score)

        print("NMF Coherence Scores:")
        pprint(nmf_coherence)
        print("NMF Semantic Scores:")
        pprint(nmf_semantic_coherence)
        print("NMF Diversity Score:")
        pprint(nmf_diversity_score)

        print("> GraphTopic...")
        model = GraphTopic(
            docs,
            max_topics=max_topics,
            TopM=TopM,
            TopN=TopN,
            overlapping_threshold=overlapping_threshold,
        )
        topics = model.get_topics()
        print("Calculating statistical coherence scores")
        m_coherence = calculate_coherence_scores(
            topics, docs, topn=max_keyphrases_per_topic
        )
        print("Calculating semantics similarity scores")
        m_semantic_coherence = get_semantic_score(embeddingModel, topics)
        print("Calculating topic diversity scores")
        m_diversity_score = calculate_topic_diversity(topics)

        print("GraphTopic Coherence Scores:")
        pprint(m_coherence)
        print("GraphTopic Semantic Scores:")
        pprint(m_semantic_coherence)
        print("GraphTopic Diversity Score:")
        pprint(m_diversity_score)

        print("---------------------------------------------------")
        break
