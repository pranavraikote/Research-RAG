import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingGenerator:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
        """

        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)  
    
    def embed_texts(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (n_texts, dimension)
        """
        
        embeddings = self.embeddings.embed_documents(texts)
        return np.array(embeddings)
    
    def embed_query(self, query):
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Numpy array of embedding (dimension,)
        """

        embedding = self.embeddings.embed_query(query)
        return np.array(embedding)