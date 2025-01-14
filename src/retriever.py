########################################################################
####               Document Assistant Using RAG                     ####
####                     KRISSAAN AMEN ALLAH                        ####        
########################################################################
####                          Retriever                             #### 
########################################################################


import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=100):
        """
        Initialize the retriever with a sentence-transformers model.
        :param model_name: Name of the pre-trained model to use.
        :param chunk_size: Number of words per chunk for large documents.
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.text_chunks = None
        self.chunk_size = chunk_size

    def preprocess(self, text):
        """
        Preprocess text by removing extra spaces and non-alphanumeric characters.
        :param text: Input text.
        :return: Cleaned text.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_document(self, text):
        """
        Split the document into smaller chunks for embedding.
        :param text: Input text.
        :return: List of chunks.
        """
        words = text.split()
        chunks = [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        return chunks

    def load_document(self, file_path):
        """
        Load and preprocess the document, then create embeddings.
        :param file_path: Path to the text document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
        
        preprocessed_text = self.preprocess(raw_text)
        self.text_chunks = self.chunk_document(preprocessed_text)
        self.embeddings = self.model.encode(self.text_chunks)

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k most relevant chunks from the document for a query.
        :param query: Query string.
        :param top_k: Number of top results to return.
        :return: List of tuples (chunk, similarity_score).
        """
        if self.embeddings is None or self.text_chunks is None:
            raise ValueError("No document loaded. Use load_document() first.")
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(self.text_chunks[i], similarities[i]) for i in top_indices]
        return results
