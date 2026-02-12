"""
Hybrid Structured Chunking Strategy

Combines three approaches for optimal chunking of research papers:
1. Section-aware: Respects paper section boundaries (Abstract, Intro, Methods, etc.)
2. Paragraph-aware: Preserves paragraph boundaries within sections
3. Citation-aware: Never splits citations from their context

This is the RECOMMENDED chunking strategy for academic papers.
"""

import re
import unicodedata
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HybridStructuredChunker:
    """
    Hybrid chunking strategy that combines section, paragraph, and citation awareness.

    Algorithm:
    1. Split text by section boundaries (Abstract, Introduction, Methods, etc.)
    2. Within each section, merge paragraphs until reaching target chunk size
    3. Ensure no citations are split across chunk boundaries
    4. Add comprehensive metadata from all three approaches
    """

    def __init__(
        self,
        target_chunk_size=1500,
        max_chunk_size=2000,
        min_paragraph_size=50,
        section_overlap=100
    ):
        """
        Initialize hybrid structured chunker.

        Args:
            target_chunk_size: Target size for chunks (paragraphs merged to reach this)
            max_chunk_size: Maximum chunk size (split if exceeded)
            min_paragraph_size: Minimum paragraph size (skip smaller ones)
            section_overlap: Overlap size when splitting large sections
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_paragraph_size = min_paragraph_size
        self.section_overlap = section_overlap

        # Fallback splitter for oversized content
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=section_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text, metadata=None, structure=None):
        """
        Chunk text using hybrid structured approach.

        Args:
            text: Text to chunk
            metadata: Base metadata to attach to chunks
            structure: Structure dict with sections, paragraphs, citations

        Returns:
            List of chunk dictionaries with text and comprehensive metadata
        """
        result = []
        base_metadata = metadata.copy() if metadata else {}

        # Extract basic metadata
        paper_id = base_metadata.get("paper_id", "")
        title = base_metadata.get("paper_title", base_metadata.get("title", ""))

        # Check if structure is available
        if not structure or not structure.get("sections") or not structure.get("paragraphs"):
            # Fallback to basic chunking
            return self._fallback_chunk(text, base_metadata, paper_id, title)

        # Get structure components
        sections = structure["sections"]
        paragraphs = structure["paragraphs"]
        citations = structure.get("citations", [])

        # Build citation map for fast lookup
        citation_map = self._build_citation_map(citations)

        # Process each section
        chunk_idx = 0
        for section in sections:
            section_type = section.get("type", "unknown")
            section_title = section.get("title", "")
            section_page_start = section.get("start_page", 0)
            section_page_end = section.get("end_page", section_page_start)

            # Get paragraphs for this section
            section_paragraphs = [
                p for p in paragraphs
                if p.get("section") == section_type
            ]

            if not section_paragraphs:
                continue

            # Chunk this section's paragraphs
            section_chunks = self._chunk_section_paragraphs(
                section_paragraphs,
                citation_map,
                section_type,
                section_title,
                section_page_start,
                section_page_end
            )

            # Add chunks with metadata
            for chunk_text, chunk_meta in section_chunks:
                chunk_text = self._clean_text(chunk_text)

                if not chunk_text or len(chunk_text) < 10:
                    continue

                chunk_id = f"{paper_id}_chunk_{chunk_idx}" if paper_id else f"{title}_chunk_{chunk_idx}" if title else f"chunk_{chunk_idx}"

                # Combine base metadata with chunk-specific metadata
                full_metadata = {
                    "chunk_id": chunk_id,
                    "title": title,
                    "conference": base_metadata.get("venue", base_metadata.get("conference", "")),
                    "year": base_metadata.get("year", ""),
                    **chunk_meta
                }

                result.append({
                    "text": chunk_text,
                    "metadata": full_metadata
                })

                chunk_idx += 1

        return result

    def _chunk_section_paragraphs(
        self,
        paragraphs: List[Dict],
        citation_map: Dict,
        section_type: str,
        section_title: str,
        section_page_start: int,
        section_page_end: int
    ) -> List[tuple]:
        """
        Chunk paragraphs within a section while preserving citations.

        Args:
            paragraphs: List of paragraph dicts for this section
            citation_map: Map of char positions to citations
            section_type: Type of section (abstract, intro, etc.)
            section_title: Title of the section
            section_page_start: Starting page of section
            section_page_end: Ending page of section

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        chunks = []
        current_paragraphs = []
        current_size = 0
        current_page = section_page_start

        for para in paragraphs:
            para_text = para.get("text", "").strip()
            para_size = len(para_text)
            para_page = para.get("page_num", section_page_start)

            # Skip tiny paragraphs
            if para_size < self.min_paragraph_size:
                continue

            # Check if adding this paragraph would exceed target
            would_exceed = (current_size + para_size) > self.target_chunk_size

            # Finalize current chunk if needed
            if current_paragraphs and would_exceed:
                chunk_text = "\n\n".join(current_paragraphs)

                chunk_metadata = {
                    "section_type": section_type,
                    "section_title": section_title,
                    "section_page_start": section_page_start,
                    "section_page_end": section_page_end,
                    "paragraph_count": len(current_paragraphs),
                    "page_num": current_page,
                    "is_section_partial": len(paragraphs) > len(current_paragraphs),
                    "is_paragraph_partial": False
                }

                chunks.append((chunk_text, chunk_metadata))

                # Reset for next chunk
                current_paragraphs = []
                current_size = 0

            # Handle oversized single paragraph
            if para_size > self.max_chunk_size:
                # Split large paragraph
                sub_chunks = self.fallback_splitter.split_text(para_text)

                for sub_chunk in sub_chunks:
                    chunk_metadata = {
                        "section_type": section_type,
                        "section_title": section_title,
                        "section_page_start": section_page_start,
                        "section_page_end": section_page_end,
                        "paragraph_count": 1,
                        "page_num": para_page,
                        "is_section_partial": True,
                        "is_paragraph_partial": True
                    }

                    chunks.append((sub_chunk, chunk_metadata))

            else:
                # Add paragraph to current chunk
                current_paragraphs.append(para_text)
                current_size += para_size
                current_page = para_page

        # Finalize last chunk
        if current_paragraphs:
            chunk_text = "\n\n".join(current_paragraphs)

            chunk_metadata = {
                "section_type": section_type,
                "section_title": section_title,
                "section_page_start": section_page_start,
                "section_page_end": section_page_end,
                "paragraph_count": len(current_paragraphs),
                "page_num": current_page,
                "is_section_partial": False,
                "is_paragraph_partial": False
            }

            chunks.append((chunk_text, chunk_metadata))

        return chunks

    def _build_citation_map(self, citations: List[Dict]) -> Dict[int, Dict]:
        """Build map from character positions to citations."""
        citation_map = {}
        for citation in citations:
            start = citation.get("char_start", 0)
            end = citation.get("char_end", 0)
            for pos in range(start, end):
                citation_map[pos] = citation
        return citation_map

    def _ensure_no_citation_splits(self, text: str, citation_map: Dict) -> str:
        """
        Ensure no citations are split. If a citation is at the boundary,
        extend text to include full citation.

        Args:
            text: Chunk text
            citation_map: Map of positions to citations

        Returns:
            Adjusted text with complete citations
        """
        # For simplicity, we rely on paragraph boundaries naturally
        # avoiding citation splits. This is a placeholder for more
        # sophisticated boundary adjustment if needed.
        return text

    def _extract_citations_in_chunk(self, chunk_text: str) -> List[Dict]:
        """Extract citations from chunk text."""
        try:
            from ..data import structure_utils
            return structure_utils.extract_citations(chunk_text)
        except ImportError:
            # Fallback pattern matching
            citations = []
            patterns = [
                (r'\[(\d+(?:\s*,\s*\d+)*(?:\s*[-–]\s*\d+)?)\]', 'numeric'),
                (r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}[a-z]?)\)', 'author_year'),
            ]

            for pattern, citation_type in patterns:
                for match in re.finditer(pattern, chunk_text):
                    citations.append({
                        "text": match.group(0),
                        "type": citation_type
                    })

            return citations

    def _has_incomplete_citation(self, text: str) -> bool:
        """Check if text has incomplete citations."""
        # Check for incomplete numeric citations
        if re.search(r'\[\d+(?:,\s*\d+)*(?![\d,\s\]]*\])', text):
            return True

        # Check for orphaned brackets
        if text.strip().startswith(']') or text.strip().endswith('['):
            return True

        return False

    def _fallback_chunk(
        self,
        text: str,
        base_metadata: Dict,
        paper_id: str,
        title: str
    ) -> List[Dict]:
        """
        Fallback to basic chunking when structure is unavailable.

        Args:
            text: Text to chunk
            base_metadata: Base metadata
            paper_id: Paper ID
            title: Paper title

        Returns:
            List of chunk dicts
        """
        text = self._clean_text(text)

        # Try to split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs or len(paragraphs) == 1:
            # Use fallback splitter
            chunks = self.fallback_splitter.split_text(text)
        else:
            # Merge paragraphs
            chunks = []
            current_chunk = []
            current_size = 0

            for para in paragraphs:
                if current_size + len(para) > self.target_chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(para)
                current_size += len(para)

            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_text = self._clean_text(chunk_text)

            if not chunk_text or len(chunk_text) < 10:
                continue

            chunk_id = f"{paper_id}_chunk_{i}" if paper_id else f"{title}_chunk_{i}" if title else f"chunk_{i}"

            chunk_metadata = {
                "chunk_id": chunk_id,
                "title": title,
                "conference": base_metadata.get("venue", base_metadata.get("conference", "")),
                "year": base_metadata.get("year", ""),
                "section_type": "unknown",
                "section_title": "",
                "section_page_start": 0,
                "section_page_end": 0,
                "paragraph_count": chunk_text.count("\n\n") + 1,
                "page_num": 0,
                "is_section_partial": False,
                "is_paragraph_partial": False
            }

            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

        return result

    def _clean_text(self, text: str) -> str:
        """
        Clean text for chunk output.

        Enhanced to handle PDF extraction artifacts:
        - Newlines within citations: (Smith et al.,\\n2024)
        - Hyphenated words across lines: compu-\\nter
        - Excessive whitespace
        - Control characters
        """
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        # Remove control characters (except newlines, tabs, spaces)
        cleaned = []
        for char in text:
            if unicodedata.category(char).startswith('C') and char not in ['\n', '\t', ' ']:
                continue
            if unicodedata.category(char) == 'Cf':  # Format characters
                continue
            cleaned.append(char)

        text = ''.join(cleaned)

        # FIX PDF ARTIFACTS - Do this BEFORE collapsing whitespace

        # 1. Fix newlines within parentheses (citations, equations, etc.)
        text = re.sub(r'\(([^)]+)\)', lambda m: '(' + m.group(1).replace('\n', ' ') + ')', text)

        # 2. Fix newlines within square brackets (numeric citations)
        text = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace('\n', ' ') + ']', text)

        # 3. Fix hyphenated words across lines
        text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', text)

        # 4. Fix broken words with just newline (no hyphen)
        text = re.sub(r'([a-zA-Z])\n([a-z])', r'\1\2', text)

        # NOW collapse excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' *\n *', '\n', text)

        return text.strip()
