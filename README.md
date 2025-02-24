# GraphTopic

## Introduction to GraphTopic

GraphTopic is a topic modeling framework designed for extracting and merging topics from a list of documents using a graph-based approach. It leverages keyphrase extraction, sentence embeddings, and graph theory to identify coherent and meaningful topics within a document collection.

This method excels at discovering nuanced topic structures by:

1. **Extracting Keyphrases:**  Utilizes a keyphrase extraction technique (defaulting to KeyBERT) to identify important phrases within documents.
2. **Building Trivial Topics:** Initializes individual topics based on top keyphrases from a merged document representing the entire corpus.
3. **Merging Latent Topics:** Iteratively processes documents, merging related trivial topics based on shared keyphrases and semantic similarity. This forms a graph of interconnected topics.
4. **Final Topic Aggregation:** Merges semantically close latent topics using sentence embeddings to produce a concise and comprehensive set of final topics.

## Instructions on How to Create and Run the Environment

To set up the environment for running GraphTopics, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dev-SR/GraphTopic.git
   ```

2. **Create a Virtual Environment**:
   It is recommended to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Required Packages**:
   Install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download required `nltk` resources**
   Go to python shell and run:
   ```python
   venv\Scripts\activate
   python
   >>> import nltk
   >>> nltk.download("stopwords")
   >>> nltk.download("punkt")
   >>> nltk.download("punkt_tab")
   >>> nltk.download("averaged_perceptron_tagger")
   >>> nltk.download("averaged_perceptron_tagger_eng")
   ```

6. **Run a sample example**:
   To execute the tests and see the results, run:
   ```bash
   python runner.py
   ```

## Usage

### Basic Example

```python
from GraphTopic import GraphTopic

# List of documents to analyze
documents = [
    "Machine learning is a field of artificial intelligence that focuses on building models....",
    "Deep learning is a subset of machine learning that uses neural networks.....",
    "Graph neural networks are used for analyzing graph-structured data...."
]

# Initialize the GraphTopic model
graph_topic_model = GraphTopic(
    docs=documents,
    TopM=300,  # Number of keyphrases to extract from the merged document
    TopN=8,    # Number of keyphrases to extract from each document
    overlapping_threshold=0.6  # Threshold for merging topics
)

# Retrieve extracted topics
topics = graph_topic_model.get_topics()
print("Extracted Topics:", topics)
```


Here's a basic example demonstrating how to use the `GraphTopic` class:


### Class `GraphTopic`

#### `__init__(self, docs, TopM=500, TopN=10, keyphrase_extractor=getKeyPhrasesUsingKeyBERT, embedding_model=embeddingModel, pretty_print=False, prune_node_count_less_than=2, overlapping_threshold=0.5, max_topics=float("inf"), max_keyphrases_per_topic=100, keyphrases_selection_strategy=KeyphrasesSelectionStrategy.CDIST)`

**Initializes the GraphTopic model.**

**Parameters:**

* **`docs` (List[str])**:  A list of document texts to be analyzed.
* **`TopM` (int, optional)**:
    * **Default:** `500`
    * **Description:**  Determines the number of top keyphrases extracted from the merged document. These keyphrases are used to initialize trivial topics. A higher `TopM` value may lead to more granular and specific initial topics.
* **`TopN` (int, optional)**:
    * **Default:** `10`
    * **Description:**  Specifies the number of top keyphrases to extract from each individual document. These document keyphrases are used to merge and refine the trivial topics into latent topics. A lower `TopN` value may result in broader, more general latent topics.
* **`keyphrase_extractor` (Callable[[str, int], List[Tuple[str, float]]], optional)**:
    * **Default:** `getKeyPhrasesUsingKeyBERT`
    * **Description:** A function responsible for extracting keyphrases from a given text. It should accept the text and the desired number of keyphrases (`n`) as input and return a list of tuples, where each tuple contains a keyphrase (string) and its associated weight (float). You can provide a custom keyphrase extraction function if needed, conforming to this signature.
    * **Example Function Signature:**
        ```python
        def getKeyPhrases(text, n):
            # ... your keyphrase extraction logic ...
            return [("keyphrase1", 0.9), ("keyphrase2", 0.8), ...]
        ```
* **`embedding_model` (SentenceTransformer, optional)**:
    * **Default:** `embeddingModel` (from `GraphTopic.helpers.config`, typically a SentenceTransformer model)
    * **Description:**  A SentenceTransformer model used for embedding keyphrases when merging semantically similar topics. This model is crucial for determining topic similarity during the final topic merging phase.
* **`pretty_print` (bool, optional)**:
    * **Default:** `False`
    * **Description:**  If set to `True`, enables verbose output during the topic merging process, printing intermediate steps and similarity scores. Useful for debugging and understanding the merging process.
* **`prune_node_count_less_than` (int, optional)**:
    * **Default:** `2`
    * **Description:**  Specifies the minimum number of keyphrases (nodes in the topic graph) that a latent topic must have to be retained. Topics with fewer keyphrases than this threshold are pruned from the latent topic graph, helping to filter out less significant or overly specific topics.
* **`overlapping_threshold` (float, optional)**:
    * **Default:** `0.5`
    * **Description:**  This is the cosine similarity threshold used to merge semantically similar latent topics into final topics. A higher `overlapping_threshold` means topics need to be more similar to be merged. Increasing this threshold will generally result in fewer, more distinct final topics.
* **`max_topics` (float, optional)**:
    * **Default:** `float("inf")` (infinity - all topics are returned)
    * **Description:**  Limits the maximum number of final topics to be returned. If set to a finite number, the method will return only the top `max_topics` topics, ranked according to the chosen `keyphrases_selection_strategy`.
* **`max_keyphrases_per_topic` (int, optional)**:
    * **Default:** `100`
    * **Description:**  Specifies the maximum number of keyphrases to include in each final topic when retrieving the topics using `get_topics()`. This is useful for controlling the size and detail of the topic representations.
* **`keyphrases_selection_strategy` (KeyphrasesSelectionStrategy, optional)**:
    * **Default:** `KeyphrasesSelectionStrategy.CDIST`
    * **Description:**  Determines the strategy used to select the top keyphrases for each topic when `max_topics` is specified. Options include:
        * `KeyphrasesSelectionStrategy.IMP_NODE`: Selects keyphrases based on their importance in the topic graph (node outdegree centrality).
        * `KeyphrasesSelectionStrategy.CDIST`: Selects keyphrases that are most semantically central to the topic, based on cosine similarity to the topic centroid embedding.
        * `KeyphrasesSelectionStrategy.RNDOM`: Randomly selects keyphrases from the topic.

#### `get_topics(self) -> List[List]`

**Retrieves the final extracted topics.**

**Returns:**

* **`List[List]`**: A list of final topics. Each topic is represented as a list of keyphrases (strings). If `max_topics` is specified in the constructor, each topic will contain at most `max_keyphrases_per_topic` keyphrases, selected according to the `keyphrases_selection_strategy`.



## Comparison of Running Tests with Other Models

The `test-runner.py` script provides a comprehensive comparison of different topic modeling techniques, including:

- **BERT Topic Modeling**: - Utilizes BERT embeddings to extract topics and evaluate their coherence, semantic similarity, and diversity.
- **LDA (Latent Dirichlet Allocation)**: A generative probabilistic model that identifies topics based on word distributions.
- **LSI (Latent Semantic Indexing)**: Uses singular value decomposition to reduce dimensionality and identify topics.
- **HDP (Hierarchical Dirichlet Process)**: A nonparametric Bayesian approach that allows for an unknown number of topics.
- **NMF (Non-negative Matrix Factorization)**: Factorizes the document-term matrix into non-negative factors to identify topics.

### Performance Metrics

For each model, the following metrics are calculated:

- **Coherence Scores**: Measure the degree of semantic similarity between high-scoring words in a topic.
- **Semantic Similarity Scores**: Evaluate how similar the topics are based on their embeddings.
- **Diversity Scores**: Assess the variety of topics generated, ensuring that they cover different aspects of the data.

By comparing these metrics across different models, users can determine which method best suits their specific dataset and analysis goals.

---

This README provides a comprehensive overview of GraphTopics, guiding users through setup, usage, and performance evaluation against other topic modeling techniques. For further assistance, please refer to the documentation or contact the maintainers.