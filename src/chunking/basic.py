import re
import json
import unicodedata
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

class BasicChunker:
    
    def __init__(self, chunk_size = 1000, chunk_overlap = 200, strategy = "recursive"):
        """
        Initialize basic chunker.
        
        Args:
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy ('recursive' or 'character')
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
                separators = ["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.splitter = CharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
                separator = "\n"
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
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters and ambiguous unicode characters
        cleaned = []
        for char in text:
            
            # Skip control characters (except common whitespace)
            if unicodedata.category(char).startswith('C') and char not in ['\n', '\t', ' ']:
                continue
            
            # Skip ambiguous unicode characters (like zero-width spaces, etc.)
            if unicodedata.category(char) == 'Cf':  # Format characters
                continue

            cleaned.append(char)
        
        text = ''.join(cleaned)
        
        # Replace newlines and multiple spaces with single spaces
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' +', ' ', text)
        
        # More cleaning: hyphen-space and other commonly found patterns
        text = re.sub(r'([a-zA-Z])-\s+([a-z])', r'\1\2', text)
        text = re.sub(r'\b\d+\.\s*\.\.\.\s*\.\.\.', '', text)
        text = re.sub(r'Type your question here\.\.\.', '', text)
        
        # The mandatory final step to strip clean the text
        text = text.strip()
        
        return text
    
    # This is something that can work brilliantly or fail miserably :P
    def _extract_title_from_text(self, text):
        """
        Extract title function.
        
        Args:
            text: Long text of the paper
        
        Returns:
            best_title: Potential title of the paper
        """

        if not text:
            return ""
        
        # Split into words
        lines = text.split('\n')
        
        # Finding where "Abstract" appears
        abstract_idx = None
        for i, line in enumerate(lines):
            if 'Abstract' in line or 'abstract' in line.lower():
                abstract_idx = i
                break
        
        # Title is always before Abstract :P
        search_range = lines[:abstract_idx] if abstract_idx else lines[:15]
        
        skip_patterns = [
            'Proceedings of',
            'Annual Meeting',
            'Association for',
            'Volume',
            'pages',
            '©',
            'Department of',
            'Department'
            '@',
            'Work done',
            'Visiting',
            'University',
            '.com',
            '.edu'
        ]
        
        best_title = ""
        best_score = 0
        
        for line in search_range:
            line_clean = self._clean_text(line).strip()
            
            # Skip empty or very short lines
            if len(line_clean) < 20:
                continue
            
            # Skip matches skip patterns
            if any(line_clean.startswith(pattern) for pattern in skip_patterns):
                continue
            
            # Skip if it's all caps and long usually header or other conference information
            if line_clean.isupper() and len(line_clean) > 50:
                continue
            
            # Scoring the candidates
            score = 0

            if 30 <= len(line_clean) <= 200:
                score += 10
            if ':' in line_clean: 
                score += 5
            if any(word[0].isupper() for word in line_clean.split()[:3]):
                score += 3
            
            if score > best_score:
                best_score = score
                best_title = line_clean
        
        return best_title
    
    def chunk(self, text, metadata = None):
        """
        Chunking function.
        
        Args:
            text: Text that is to be chunked
            metadata: Metadata to attach to each chunk
            
        Returns:
            result: List of chunk dictionaries with text and metadata
        """

        result = []
        base_metadata = metadata.copy() if metadata else {}

        # Extracting and setting some fields that have already been captured
        paper_id = base_metadata.get("paper_id", "")
        title = base_metadata.get("paper_title", base_metadata.get("title", ""))
        
        # If missing, call our function and hope it works
        if not title or title.strip() == "":
            title = self._extract_title_from_text(text)
        
        # Mandatory cleaning before chunking
        text = self._clean_text(text)
        
        # The first pass chunking process
        chunks = self.splitter.split_text(text)
        
        # This is responsible for the metadata :)
        for i, chunk_text in enumerate(chunks):
            
            # Second pass of cleaning, although not required, there could be something left out, and this is anyways super fast
            chunk_text = self._clean_text(chunk_text)
            
            # Skip empty chunks
            if not chunk_text or len(chunk_text.strip()) < 10:
                continue
            
            # Setting the chunk_id as paper_id_chunk_id or title_chunk_id or chunk_id, auto increment the idx
            chunk_id = f"{paper_id}_chunk_{i}" if paper_id else f"{title}_chunk_{i}" if title else f"chunk_{i}"
            
            # Putting things together here
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
    
    # This was developed later on after adjusting the structure of the chunks, retaining both versions for usability
    def save_chunks(self, chunks, chunks_path):
        """
        Saving chunking function.
        
        Args:
            chunks: List of chunk dictionaries
            chunks_path: Path to save chunks JSON file
        """

        result = []
        chunk_id = 1
        
        # Iterating through the chunks, preparing the blocks for final saving
        for chunk in chunks:
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "title": chunk["metadata"].get("title", ""),
                "text": chunk["text"],
                "conference": chunk["metadata"].get("conference", ""),
                "year": chunk["metadata"].get("year", "")
            }

            result.append(chunk_dict)
            chunk_id += 1
        
        # Configuring the path and saving the chunks.json to disk
        Path(chunks_path).parent.mkdir(parents = True, exist_ok = True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_chunks(chunks_path):
        """
        Loading chunks function.
        
        Args:
            chunks_path: Path to chunks JSON file
            
        Returns:
            (chunk_texts, chunk_metadata) tuple
        """

        # Opening the chunks.json file
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        
        chunk_texts = []
        chunk_metadata = []
        
        # Reading the data and splitting it for later use
        for chunk in chunks_data:
            chunk_texts.append(chunk["text"])
            chunk_metadata.append({
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
                "conference": chunk["conference"],
                "year": chunk["year"]
            })
        
        return chunk_texts, chunk_metadata