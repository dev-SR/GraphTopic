from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

reducer = PCA(n_components=3, random_state=45)
model_path_or_name = "sentence-transformers/all-mpnet-base-v2"
embeddingModel = SentenceTransformer(model_path_or_name)
th = 0.5
