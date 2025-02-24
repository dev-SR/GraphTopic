from gensim.models import LdaModel, LsiModel, HdpModel, Nmf
from gensim.matutils import Sparse2Corpus
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary


def preprocess_gensim(texts, ngram_range=(2, 3)):
    # Create and fit CountVectorizer with desired n-gram range
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
    )
    corpus_vect = vectorizer.fit_transform(texts)
    corpus_vect_gensim = Sparse2Corpus(corpus_vect, documents_columns=False)
    dictionary = Dictionary.from_corpus(
        corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()),
    )
    return corpus_vect_gensim, dictionary


def get_lda_topics(corpus, dictionary, num_topics=10, num_words=10):
    print("> LDA...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        random_state=40,
    )
    topics = []
    for topic_id, topic in lda_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics


def get_lsi_topics(corpus, dictionary, num_topics=10, num_words=10):
    print("> LSI...")
    lsi_model = LsiModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, random_seed=40
    )
    topics = []
    for topic_id, topic in lsi_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics


def get_hdp_topics(corpus, dictionary, num_topics=10, num_words=10):
    print("> HDP...")
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary, random_state=40)
    topics = []
    for topic_id, topic in hdp_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics


def get_nmf_topics(corpus, dictionary, num_topics=10, num_words=10):
    print("> NMF...")
    nmf_model = Nmf(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=40
    )
    topics = []
    for topic_id, topic in nmf_model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    ):
        topic = [word for word, _ in topic]
        topics.append(topic)
    return topics


def get_bertopic_domains_with_kps(docs, words_per_topic=10):
    print("> Bertopic...")
    if len(docs) <= 5:
        docs *= 5  # Duplicate the docs if its length is less than 5

    cv = CountVectorizer(ngram_range=(2, 3), stop_words="english")
    try:
        topic_model = BERTopic(
            language="english",
            top_n_words=words_per_topic,
            vectorizer_model=cv,
            min_topic_size=2,
        )
        topic_model.fit_transform(docs)
    except Exception as e:
        print(e)
        return None
    topics_res = topic_model.get_topics()
    bert_topics = []
    for topic_idx, topic_keywords in topics_res.items():
        keywords = [keyword for keyword, _ in topic_keywords]
        bert_topics.append(keywords)
    return bert_topics
