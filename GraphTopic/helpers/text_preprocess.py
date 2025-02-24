import re
from typing import List
import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("averaged_perceptron_tagger")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

REPLACE_PUNCTUATION_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,:;]")  # exclude full stop
REMOVE_NUM = re.compile("[\d+]")
EMAIL_RE = re.compile("\b[\w\-.]+?@\w+?\.\w{2,4}\b")
PHONE_RE = re.compile("\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b")
NUMBER_RE = re.compile("\d+(\.\d+)?")
URLS_RE = re.compile("(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)")
EXTRA_SPACE_RE = re.compile("\s+")
STOPWORDS = set(stopwords.words("english"))
# BAD_SYMBOLS: except numbers,character,space,newline,.,_
BAD_SYMBOLS_RE = re.compile("[^0-9a-zA-Z _.\n]")
HYPHENATED_RE = re.compile("-\n")
SPACE_BEFORE_FULLSTOPS = re.compile("\s\.")
MORE_THAN_ONE_FULLSTOPS_RE = re.compile("\.\.+")


def cleanText(text, stem=True):
    text = HYPHENATED_RE.sub("", text)  # joining Hyphenated words
    text = REPLACE_PUNCTUATION_BY_SPACE_RE.sub(" ", text)
    text = NUMBER_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = URLS_RE.sub("", text)
    text = PHONE_RE.sub("", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    text = EXTRA_SPACE_RE.sub(" ", text)

    # Lower case
    text = text.lower()

    # Remove Stop Words
    # words = [w for w in words if not w in STOPWORDS]

    if stem:
        # Tokenize
        words = word_tokenize(text)
        # # Stemming
        stemmed_words = [ps.stem(w) for w in words]
        # # Join the words back into one string separated by space,
        stemmed_sen = " ".join(stemmed_words)
        # removing space before full stops
        stemmed_sen = SPACE_BEFORE_FULLSTOPS.sub(".", stemmed_sen)
        # removing more than one full stop
        stemmed_sen = MORE_THAN_ONE_FULLSTOPS_RE.sub("", stemmed_sen)
        return stemmed_sen
    else:
        # removing space before full stops
        text = SPACE_BEFORE_FULLSTOPS.sub(".", text)
        # removing more than one full stop
        text = MORE_THAN_ONE_FULLSTOPS_RE.sub("", text)
        return text


def extract_noun_phrases(text):
    text = cleanText(text, stem=False)
    # print(text)
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Initialize a list to store the noun phrases for each sentence
    noun_phrases_list = []
    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words and tag them with their part of speech
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        # Define a grammar for noun phrases
        # grammar = r'NP: {<DT>?<JJ>*<NN>}'
        # grammar = r'NP: {<DT>?<JJ>*<NNS|NN|NNP|NNPS|PRP$>+}'
        # grammar = r'NP: {<NNS|NN|NNP|NNPS|PRP$>+}'
        # grammar = r'NP: {(<NN.*>+<JJ.*>?)|(<JJ.*>?<NN.*>+)}'
        grammar = r"NP: {(<NN.*>?)|(<NN.*>+)}"

        chunk_parser = nltk.RegexpParser(grammar)
        chunked_pos_tags = chunk_parser.parse(pos_tags)
        # Extract the noun phrases from the chunked POS tags
        noun_phrases = []
        for subtree in chunked_pos_tags.subtrees():
            if subtree.label() == "NP":
                noun_phrase = " ".join([word for (word, pos) in subtree])
                noun_phrases.append(noun_phrase)
        # Append the list of noun phrases for the current sentence to the list
        noun_phrases_list.append(noun_phrases)
        noun_phrases_list.append(".")
    # Flatten the list of lists of noun phrases into a single list
    flattened_noun_phrases = [
        noun_phrase for sublist in noun_phrases_list for noun_phrase in sublist
    ]
    # Join the noun phrases into a single string with full stops
    return " ".join(flattened_noun_phrases).replace(" .", ".")


def split_into_chunks(raw_docs, chunk_size, overlap_ratio=0):
    chunked_documents = []
    overlap = int(chunk_size * overlap_ratio)

    for i in range(0, len(raw_docs), chunk_size - overlap):
        chunk = raw_docs[i : i + chunk_size]
        # print(i, i + chunk_size)
        chunked_documents.append(chunk)

    # print()
    # print(len(chunked_documents))

    return chunked_documents


def _preprocess_text(documents, deep_clean=False) -> List[str]:
    """Basic preprocessing of text

    Steps:
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
        * Apply PorterStemmer for word stemming (optional)
        * Remove stopwords
        * Remove documents with less than 5 words
    """
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    if deep_clean:
        cleaned_documents = [
            re.sub(r"[^A-Za-z0-9 . -]+", "", doc) for doc in cleaned_documents
        ]
        cleaned_documents = [
            doc if doc != "" else "emptydoc" for doc in cleaned_documents
        ]

    # Remove documents with less than 5 words
    cleaned_documents = [doc for doc in cleaned_documents if len(doc.split()) >= 5]

    return cleaned_documents
