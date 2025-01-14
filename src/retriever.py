import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a sentence-transformers model.
        :param model_name: Name of the pre-trained model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.text_data = None

    def load_document(self, file_path):
        """
        Load and split the document into sentences.
        :param file_path: Path to the text document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text_data = file.readlines()
        
        # Remove empty lines
        self.text_data = [line.strip() for line in self.text_data if line.strip()]
        self.embeddings = self.model.encode(self.text_data)

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k most relevant sentences from the document for a query.
        :param query: Query string.
        :param top_k: Number of top results to return.
        :return: List of tuples (sentence, similarity_score).
        """
        if self.embeddings is None or self.text_data is None:
            raise ValueError("No document loaded. Use load_document() first.")
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.text_data[i], similarities[i]) for i in top_indices]
