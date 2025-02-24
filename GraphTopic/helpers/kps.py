from keybert import KeyBERT


def getKeyPhrasesUsingKeyBERT(text="", n=15):
    kw_model = KeyBERT()
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 3),
        stop_words="english",
        top_n=n,
        # diversity=0.5,
    )
