import re
import numpy as np
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    
    def __init__(self, embedding_generator, similarity_threshold = 0.7,
        min_chunk_size = 350, max_chunk_size = 1500, segment_size = 100):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for computing embeddings
            similarity_threshold: Threshold for determining chunk boundaries (0-1, higher = stricter)
            min_chunk_size: Minimum size of chunks
            max_chunk_size: Maximum size of chunks
            segment_size: Size of segments to compare (smaller = more granular boundaries)
        """
        
        if embedding_generator is None:
            raise ValueError("You have not given me the embedding_generator instance :(")
        
        self.embedding_generator = embedding_generator
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.segment_size = segment_size
        
        # Use a simple splitter to create initial segments for comparison
        self.segment_splitter = RecursiveCharacterTextSplitter(
            chunk_size = segment_size,
            chunk_overlap = 0,
            separators = ["\n\n", "\n", ". ", " ", ""]
        )
    
    def _clean_text(self, text):
        """
        Cleaning function.
        
        Args:
            text: Un-cleaned text
        
        Returns:
            text: Cleaned text
        """
        
        if not text:
            return ""
        
        # Basic cleaning only, we'll let the nuances and everything else as-is and let the semantics be handled
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def _compute_similarities(self, embeddings):
        """
        Cosine similarity function.
        
        Args:
            embeddings: Numpy array of embeddings (n_segments, dimension)
        
        Returns:
            similarities: Array of similarity scores between adjacent segments
        """
        
        # Checking the embedding dims before proceeding
        if len(embeddings) < 2:
            return np.array([])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis = 1, keepdims = True)
        norms[norms == 0] = 1 
        normalized = embeddings / norms
        
        # Computing the dot products
        similarities = []
        for i in range(len(normalized) - 1):
            sim = np.dot(normalized[i], normalized[i + 1])
            similarities.append(float(sim))
        
        return np.array(similarities)
    
    def _find_boundaries(self, segments, similarities):
        """
        Chunk boundaries function.
        
        Args:
            segments: List of segment texts
            similarities: Array of similarity scores between adjacent segments
        
        Returns:
            boundaries: List of indices where chunks should be split
        """
        
        # Starting with first segment
        boundaries = [0]  
        
        # Marking boundaries based on the sim thresholds
        for i, sim in enumerate(similarities):
            
            # If similarity drops below threshold, mark as boundary
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        
        # Always end at the last segment
        if boundaries[-1] != len(segments):
            boundaries.append(len(segments))
        
        return boundaries
    
    def _merge_segments_into_chunks(self, segments, boundaries):
        """
        Merge segments function.
        
        Args:
            segments: List of segment texts
            boundaries: List of boundary indices
        
        Returns:
            chunks: List of chunk texts
        """
        
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Merge segments in this range
            chunk_text = " ".join(segments[start_idx:end_idx])
            chunk_text = self._clean_text(chunk_text)
            
            # Skip if too small
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            # Split if too large
            if len(chunk_text) > self.max_chunk_size:
                
                # Split the large chunk roughly in half
                mid = len(chunk_text) // 2
                
                # Try to find a sentence boundary near the middle
                split_point = mid
                for offset in [-100, 0, 100]:
                    candidate = mid + offset
                    if 0 < candidate < len(chunk_text) and chunk_text[candidate] in ['.', '\n']:
                        split_point = candidate + 1
                        break
                
                # Split at the found point (or mid if no boundary found)
                chunks.append(self._clean_text(chunk_text[:split_point]))
                remaining = self._clean_text(chunk_text[split_point:])
                
                # Only add remaining if it's large enough
                if len(remaining) >= self.min_chunk_size:
                    chunks.append(remaining)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk(self, text, metadata = None):
        """
        Chunking function.
        
        Args:
            text: Text that is to be chunked
            metadata: Metadata to attach to each chunk
        
        Returns:
            result: List of chunk dictionaries with text and metadata
        """
        
        if not text:
            return []
        
        result = []
        base_metadata = metadata.copy() if metadata else {}
        
        # Extract metadata fields (same structure as BasicChunker)
        paper_id = base_metadata.get("paper_id", "")
        title = base_metadata.get("paper_title", base_metadata.get("title", ""))
        
        # Mandatory cleaning
        text = self._clean_text(text)
        
        if len(text) < self.min_chunk_size:
            
            # Text is too short, return as single chunk
            chunk_id = f"{paper_id}_chunk_0" if paper_id else f"{title}_chunk_0" if title else "chunk_0"
            result.append({
                "text": text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": title,
                    "conference": base_metadata.get("venue", base_metadata.get("conference", "")),
                    "year": base_metadata.get("year", ""),
                }
            })
            return result
        
        # Split text into small segments for comparison
        segments = self.segment_splitter.split_text(text)
        segments = [self._clean_text(s) for s in segments if s.strip()]
        
        if len(segments) < 2:
            
            # Not enough segments to compute, return as single chunk
            chunk_id = f"{paper_id}_chunk_0" if paper_id else f"{title}_chunk_0" if title else "chunk_0"
            result.append({
                "text": text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "title": title,
                    "conference": base_metadata.get("venue", base_metadata.get("conference", "")),
                    "year": base_metadata.get("year", ""),
                }
            })
            return result
        
        # Getting embeddings and computing similarities for the segments
        embeddings = self.embedding_generator.embed_texts(segments)
        similarities = self._compute_similarities(embeddings)
        
        # Finding boundaries where similarity drops
        boundaries = self._find_boundaries(segments, similarities)
        
        # Now, merging segments into chunks respecting size constraints
        chunk_texts = self._merge_segments_into_chunks(segments, boundaries)
        
        # Preparing final chunks with metadata (same format as BasicChunker)
        for i, chunk_text in enumerate(chunk_texts):
            
            # Skip empty chunks
            if not chunk_text or len(chunk_text.strip()) < 10:
                continue
            
            # Setting the chunk_id (same format as BasicChunker)
            chunk_id = f"{paper_id}_chunk_{i}" if paper_id else f"{title}_chunk_{i}" if title else f"chunk_{i}"
            
            # Putting things together here (same structure as BasicChunker)
            chunk_metadata = {
                "chunk_id": chunk_id,
                "title": title,
                "conference": base_metadata.get("venue", base_metadata.get("conference", "")),
                "year": base_metadata.get("year", ""),
            }
            
            # Preparing the final chunk block
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return result