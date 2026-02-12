import fitz
from pathlib import Path
from typing import Dict, List, Optional
from . import structure_utils

class PDFLoader:
    
    def __init__(self):
        """
        Initialize PDF loader.
        """
        pass
    
    def load_pdf(self, pdf_path):
        """
        Loading PDF and extracting text and metadata function.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            result: Dictionary with text, metadata, and pages
        """
        
        pdf_path = Path(pdf_path)
        return self._load_with_pymupdf(pdf_path)
    
    def _load_with_pymupdf(self, pdf_path):
        """
        Loading PDF using PyMuPDF function.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            result: Dictionary with text, metadata, and pages
        """
        
        doc = fitz.open(pdf_path)
        
        text_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extracting text with better formatting
            text = page.get_text("text")
            
            if text:
                # Cleaning up the extracted text
                text = self._clean_pdf_text(text)
                text_pages.append(text)
            else:
                text_pages.append("")
        
        # Getting metadata
        metadata_dict = doc.metadata
        
        metadata = {
            "title": metadata_dict.get("title", ""),
            "author": metadata_dict.get("author", ""),
            "subject": metadata_dict.get("subject", ""),
            "creator": metadata_dict.get("creator", ""),
            "producer": metadata_dict.get("producer", ""),
            "creation_date": metadata_dict.get("creationDate", ""),
            "modification_date": metadata_dict.get("modDate", ""),
            "num_pages": len(doc)
        }
        
        doc.close()
        
        return {
            "text": "\n\n".join(text_pages),
            "metadata": metadata,
            "pages": text_pages
        }

    def load_pdf_with_structure(self, pdf_path):
        """
        Load PDF with enhanced structure detection (sections, citations, paragraphs).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with:
                - text: Full text
                - pages: List of page texts
                - metadata: PDF metadata
                - structure: Dict with sections, paragraphs, citations
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)

        # Extract text blocks with layout information
        blocks = []
        text_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get structured text (dict mode includes font info, bboxes)
            page_dict = page.get_text("dict")

            page_text_blocks = []

            # Process each block in the page
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                block_info = {
                                    "text": text,
                                    "font_size": span.get("size", 10.0),
                                    "font": span.get("font", ""),
                                    "bbox": span.get("bbox", (0, 0, 0, 0)),
                                    "page_num": page_num
                                }
                                blocks.append(block_info)
                                page_text_blocks.append(text)

            # Store page text
            page_text = " ".join(page_text_blocks)
            if page_text:
                page_text = self._clean_pdf_text(page_text)
            text_pages.append(page_text)

        # Extract PDF metadata
        metadata_dict = doc.metadata
        metadata = {
            "title": metadata_dict.get("title", ""),
            "author": metadata_dict.get("author", ""),
            "subject": metadata_dict.get("subject", ""),
            "creator": metadata_dict.get("creator", ""),
            "producer": metadata_dict.get("producer", ""),
            "creation_date": metadata_dict.get("creationDate", ""),
            "modification_date": metadata_dict.get("modDate", ""),
            "num_pages": len(doc)
        }

        doc.close()

        # Build full text
        full_text = "\n\n".join(text_pages)

        # Detect structure using utility functions
        sections = structure_utils.detect_section_boundaries(blocks)

        # Extract full text for each section
        for section in sections:
            section["text"] = structure_utils.extract_section_text(section, blocks)
            section["end_page"] = blocks[section["end_idx"]]["page_num"] if section["end_idx"] < len(blocks) else section["start_page"]
            section["char_start"] = 0  # Will be computed if needed
            section["char_end"] = len(section["text"])

        # Extract citations from full text
        citations = structure_utils.extract_citations(full_text)

        # Identify paragraphs
        paragraphs = structure_utils.identify_paragraphs(blocks, spacing_threshold=1.5)

        # Assign paragraphs to sections
        paragraphs = structure_utils.assign_paragraphs_to_sections(paragraphs, sections, blocks)

        # Add char_start and char_end to paragraphs
        char_offset_map = structure_utils.build_char_offset_map(full_text, blocks)
        for para in paragraphs:
            if para["block_indices"]:
                first_block = para["block_indices"][0]
                last_block = para["block_indices"][-1]
                para["char_start"] = char_offset_map.get(first_block, 0)
                para["char_end"] = char_offset_map.get(last_block, 0) + len(blocks[last_block].get("text", ""))

        # Build structure dict
        structure = {
            "sections": sections,
            "paragraphs": paragraphs,
            "citations": citations
        }

        # Validate structure and get warnings
        is_valid, warnings = structure_utils.validate_structure(structure)
        structure["is_valid"] = is_valid
        structure["warnings"] = warnings

        return {
            "text": full_text,
            "pages": text_pages,
            "metadata": metadata,
            "structure": structure
        }

    def _clean_pdf_text(self, text):
        """
        Cleaning text extracted from PDF function.

        Enhanced to handle PDF extraction artifacts:
        - Newlines within citations: (Smith et al.,\\n2024)
        - Hyphenated words across lines: compu-\\nter
        - Broken words across lines
        - Excessive whitespace

        Args:
            text: Raw text from PDF

        Returns:
            result: Cleaned text
        """

        import re

        if not text:
            return ""

        # FIX PDF ARTIFACTS FIRST

        # 1. Fix newlines within parentheses (citations, equations, etc.)
        # This is critical for citation detection: (Smith et al.,\n2024) -> (Smith et al., 2024)
        text = re.sub(r'\(([^)]+)\)', lambda m: '(' + m.group(1).replace('\n', ' ') + ')', text)

        # 2. Fix newlines within square brackets (numeric citations)
        # [1,\n2, 3] -> [1, 2, 3]
        text = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace('\n', ' ') + ']', text)

        # 3. Fix hyphenated words across lines (must be before other newline fixes)
        # compu-\nter -> computer
        text = re.sub(r'([a-zA-Z])-\s*\n\s*([a-z])', r'\1\2', text)

        # 4. Fix broken words with just newline (no hyphen)
        # Transfor\nmer -> Transformer
        text = re.sub(r'([a-zA-Z])\n([a-z])', r'\1\2', text)

        # NOW do general cleaning

        # Replacing multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Removing standalone numbers/artifacts
        text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)

        # Cleaning up multiple spaces
        text = re.sub(r' +', ' ', text)

        # Fix spaces around newlines
        text = re.sub(r' *\n *', '\n', text)

        return text.strip()