import faiss
import numpy as np
from pathlib import Path
import math
from .bm25 import BM25Retriever
from .bm25 import BM25Retriever

class SemanticRetriever:
    
    def __init__(self, index_path = None, chunks_path = None, dimension = None, metric = "IP"):
        """
        Initialize semantic retriever.
        
        Args:
            index_path: Path to saved FAISS index
            chunks_path: Path to chunks JSON file 
            dimension: Dimension of embeddings 
            metric: Distance metric ("IP" for cosine similarity, "L2" for Euclidean distance)
        """

        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata = []
        self.metric = metric.upper()
        
        # Loading the index and chunks
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
            self.dimension = self.index.d
            
            if chunks_path is None:
                chunks_path = str(Path(index_path).parent / "chunks.json")
            
            if Path(chunks_path).exists():
                self.load_chunks(chunks_path)
    
    def add_chunks(self, embeddings, chunks, metadata):
        """
        Adding chunks to the index function.
        
        Args:
            embeddings: Numpy array of embeddings (n_chunks, dimension)
            chunks: List of chunk texts
            metadata: List of metadata dictionaries
        """

        if self.index is None:
            self.dimension = embeddings.shape[1]
            
            if self.metric == "IP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

        elif embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch :(")
        
        embeddings = embeddings.astype('float32')

        # Normalizing the vectors if IP chosen
        if self.metric == "IP":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1 
            embeddings = embeddings / norms
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
    
    def load_chunks(self, chunks_path):
        """
        Load chunks and build index function.
        
        Args:
            chunks_path: Path to chunks JSON file
        """
        
        # Loading chunks from disk
        from ..chunking.basic import BasicChunker
        chunk_texts, chunk_metadata = BasicChunker.load_chunks(chunks_path)
        
        self.chunks = chunk_texts
        self.metadata = chunk_metadata
    
    def search(self, query_embedding, top_k = 10, filters = None):
        """
        Searching for relevant chunks using FAISS index function.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Dictionary of metadata filters:
                - conference: str
                - year: int
                - title: str
        
        Returns:
            results: List of retrieved chunks with scores and metadata
        """

        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # If filters are provided, then let's extend the search a bit more to get more candidates for filtering
        search_k = top_k * 3 if filters else top_k
        
        # Creating a temporary BM25Retriever instance for filter matching (if filters are used)
        filter_helper = BM25Retriever() if filters else None
        
        if self.metric == "IP":
            
            # We need to normalize the query first
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Searching for closest matches!
            similarities, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.chunks):
                    
                    # Applying the metadata filter
                    if filters and not filter_helper.matches_filter(self.metadata[idx], filters):
                        continue
                    
                    # Preparing the payload
                    results.append({
                        "text": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": similarity,
                        "rank": len(results) + 1
                    })
                    
                    # Stop if we have enough results
                    if len(results) >= top_k:
                        break
        else:
            # For L2 index (Euclidean distance)
            distances, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):
                    
                    # Applying the metadata filter
                    if filters and not filter_helper.matches_filter(self.metadata[idx], filters):
                        continue
                    
                    # L2 distance (lower is better)
                    results.append({
                        "text": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "score": float(distance),
                        "rank": len(results) + 1
                    })
                    
                    # Stop if we have enough results
                    if len(results) >= top_k:
                        break
        
        return results
    
    def save_index(self, path):
        """
        Saving FAISS index to disk function.
        
        Args:
            path: Path to save the index
        """

        faiss.write_index(self.index, path)
    
    def load_index(self, path):
        """
        Loading FAISS index from disk function.
        
        Args:
            path: Path to load the index from
        """
        
        # Loading the index from disk
        self.index = faiss.read_index(path)
        
        # Detecting metric type from loaded index
        if hasattr(self.index, 'metric_type'):
            if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                self.metric = "IP"
            elif self.index.metric_type == faiss.METRIC_L2:
                self.metric = "L2"