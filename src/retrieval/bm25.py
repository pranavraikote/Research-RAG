import nltk
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

class BM25Retriever:
    
    def __init__(self, chunks_path = None):
        """
        Initialize BM25 retriever.
        
        Args:
            chunks_path: Path to chunks JSON file
        """

        self.bm25 = None
        self.chunks = []
        self.metadata = []
        self.tokenized_chunks = []
        
        if chunks_path and Path(chunks_path).exists():
            self.load_chunks(chunks_path)
    
    def add_chunks(self, chunks, metadata):
        """
        Adding chunks function.
        
        Args:
            chunks: New chunks to be added
            metadata: New metadata to be added along with the chunks
        """

        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        # Tokenizing the text
        tokenized = [word_tokenize(chunk.lower()) for chunk in chunks]
        self.tokenized_chunks.extend(tokenized)
        
        # Rebuilding the BM25 index with the updated chunks
        self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def load_chunks(self, chunks_path):
        """
        Load chunks and build BM25 index function.
        
        Args:
            chunks: New chunks to be added
            chunks_path: Path to chunks JSON file (optional)
        """
        
        # Loading chunks from disk
        from ..chunking.basic import BasicChunker
        chunk_texts, chunk_metadata = BasicChunker.load_chunks(chunks_path)

        self.chunks = chunk_texts
        self.metadata = chunk_metadata
        
        # Building the BM25 index
        if self.chunks:
            self.tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in self.chunks]
            self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def matches_filter(self, metadata, filters):
        """
        Metadata matching the filter function.
        
        Args:
            metadata: Chunk metadata dictionary
            filters: Filter dictionary with:
                - conference: str
                - year: int
                - title: str
        
        Returns:
            bool: True if metadata matches filter criteria, False otherwise
        """
        
        if not filters:
            return True
        
        # Conference filter
        if "conference" in filters:
            filter_conf = filters["conference"]
            chunk_conf = metadata.get("conference", "").upper()
            
            if isinstance(filter_conf, str):
                if chunk_conf != filter_conf.upper():
                    return False
            else:
                if chunk_conf not in [c.upper() for c in filter_conf]:
                    return False
        
        # Year filter
        if "year" in filters:
            filter_year = filters["year"]
            chunk_year = metadata.get("year")
            
            if chunk_year is None:
                return False
            
            if isinstance(filter_year, int):
                if chunk_year != filter_year:
                    return False

            elif isinstance(filter_year, (list, tuple)):
                if chunk_year not in filter_year:
                    return False

            elif isinstance(filter_year, dict):
                
                # Range filter: {"min": 2020, "max": 2024}
                if "min" in filter_year and chunk_year < filter_year["min"]:
                    return False
                if "max" in filter_year and chunk_year > filter_year["max"]:
                    return False
        
        # Title filter, not a full-fledged match, just for partial match
        if "title" in filters:
            filter_title = filters["title"]
            chunk_title = metadata.get("title", "").lower()
            
            if isinstance(filter_title, str):
                if filter_title.lower() not in chunk_title:
                    return False
            else:
                if not any(ft.lower() in chunk_title for ft in filter_title):
                    return False
        
        return True
    
    def search(self, query, top_k =10, filters = None):
        """
        Searching for relevant chunks using BM25 index function.
        
        Args:
            query: Query string
            top_k: Number of results to return
            filters: Dictionary of metadata filters:
                - conference: str
                - year: int
                - title: str
        
        Returns:
            results: List of retrieved chunks with scores and metadata
        """

        if self.bm25 is None or len(self.chunks) == 0:
            return []
        
        # Tokenizing the query and calculating the crose wrt bm25 index
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # If filters are provided, then let's extend the search a bit more to get more candidates for filtering
        search_k = top_k * 3 if filters else top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:search_k]
        
        results = []
        for idx in top_indices:
            
            # Applying the metadata filter
            if not self.matches_filter(self.metadata[idx], filters):
                continue
            
            results.append({
                "text": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        return results