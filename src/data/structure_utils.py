"""
Utilities for detecting structure in research papers (sections, citations, paragraphs).
"""

import re
from typing import List, Dict, Tuple, Optional


# Section detection patterns (case-insensitive)
SECTION_PATTERNS = {
    'abstract': r'^(abstract|summary)\s*$',
    'introduction': r'^(\d+\.?\s*)?introduction\s*$',
    'related_work': r'^(\d+\.?\s*)?(related|previous)\s*work\s*$',
    'background': r'^(\d+\.?\s*)?(extended\s+)?background\s*$',
    'methods': r'^(\d+\.?\s*)?(method|methodology|approach|experimental\s+setup)\s*$',
    'experiments': r'^(\d+\.?\s*)?(experiment|experimental\s+results|evaluation|setup)\s*$',
    'results': r'^(\d+\.?\s*)?results?\s*(and\s+discussion)?\s*$',
    'discussion': r'^(\d+\.?\s*)?discussion\s*$',
    'limitations': r'^(\d+\.?\s*)?limitations?\s*$',
    'conclusion': r'^(\d+\.?\s*)?(conclusion|concluding\s+remarks?|summary)\s*$',
    'references': r'^(\d+\.?\s*)?(references|bibliography)\s*$',
    'appendix': r'^(\d+\.?\s*)?appendi(x|ces)\s*$',
    'acknowledgments': r'^(\d+\.?\s*)?acknowledg[e]?ments?\s*$',
}

# Citation patterns for ACL/NLP conference formats
# Handles spaces around punctuation: ( Shi et al. , 2024 )
CITATION_PATTERNS = [
    # Numeric citations: [1], [1,2], [1-3], [1, 2, 3]
    (r'\[\s*\d+(?:\s*,\s*\d+)*(?:\s*[-–]\s*\d+)?\s*\]', 'numeric'),

    # Parenthetical author-year citations with flexible spacing
    # Matches: (Smith et al., 2020), ( Shi et al. , 2024 ), etc.
    # Handles:
    #   - Spaces after ( and before )
    #   - Spaces around commas and semicolons
    #   - Multiple citations: ( A, 2020 ; B, 2021 )
    #   - Letter suffixes: 2024a, 2024b
    (r'\(\s*[A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z]+))?\s*,\s*\d{4}[a-z]?(?:\s*[;,]\s*(?:[A-Z][A-Za-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z]+))?\s*,\s*)?\d{4}[a-z]?)*\s*\)', 'author_year'),

    # Inline citations: Smith et al. (2020) or Jones (2020)
    # Handles spaces inside parentheses
    (r'[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?\s+\(\s*\d{4}[a-z]?\s*\)', 'author_year_inline'),
]


def classify_section_type(text: str) -> Optional[str]:
    """
    Classify a text string as a section type using regex patterns.

    Args:
        text: Section heading text (normalized)

    Returns:
        Section type (abstract, introduction, etc.) or None if no match
    """
    text_norm = text.strip().lower()

    for section_type, pattern in SECTION_PATTERNS.items():
        if re.match(pattern, text_norm, re.IGNORECASE):
            return section_type

    return None


def detect_section_boundaries(blocks: List[Dict], base_font_size: float = None) -> List[Dict]:
    """
    Detect section boundaries in PDF text blocks using font size heuristics and regex patterns.

    Args:
        blocks: List of text blocks from PyMuPDF dict output
                Each block: {"text": str, "font_size": float, "page_num": int, "bbox": tuple}
        base_font_size: Base body text font size (auto-detected if None)

    Returns:
        List of section dicts: [{"title": str, "type": str, "start_idx": int, "end_idx": int, ...}]
    """
    if not blocks:
        return []

    # Auto-detect base font size if not provided (median font size)
    if base_font_size is None:
        font_sizes = [b.get("font_size", 10.0) for b in blocks]
        font_sizes.sort()
        base_font_size = font_sizes[len(font_sizes) // 2]

    # Threshold for section headers (typically 1.2-2x larger than body)
    # Increased from 1.15 to 1.3 to be less aggressive
    header_threshold = base_font_size * 1.3

    sections = []
    current_section = None

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        font_size = block.get("font_size", base_font_size)

        if not text:
            continue

        # Check if this looks like a section header
        is_header = False

        # Skip very short text (< 3 chars, likely noise)
        if len(text) < 3:
            continue

        # PRIMARY: Pattern matching (the right way!)
        section_type = classify_section_type(text)

        # Heuristic 1: Matches known section pattern → SECTION HEADER
        # Font size is optional hint, pattern match is definitive
        if section_type and section_type != "unknown":
            # Pattern match found! Check if text looks reasonable for header
            if len(text) < 150:  # Headers are typically short
                is_header = True

        # Heuristic 2: Very large font (2.5x+ body) without pattern
        # Only as fallback for unusual section names we didn't anticipate
        elif font_size >= base_font_size * 2.5 and len(text) < 100:
            is_header = True

        if is_header:
            # Finalize previous section
            if current_section:
                current_section["end_idx"] = i - 1
                sections.append(current_section)

            # Start new section
            section_type = classify_section_type(text) or "unknown"
            current_section = {
                "title": text,
                "type": section_type,
                "start_idx": i,
                "start_page": block.get("page_num", 0),
                "font_size": font_size
            }

    # Finalize last section
    if current_section:
        current_section["end_idx"] = len(blocks) - 1
        sections.append(current_section)

    return sections


def extract_citations(text: str) -> List[Dict]:
    """
    Extract citations from text using multiple patterns.

    Args:
        text: Full text to search for citations

    Returns:
        List of citation dicts: [{"text": str, "type": str, "char_start": int, "char_end": int}]
    """
    # Pre-clean: Remove newlines within parentheses (common in PDF extraction)
    # This helps match citations like "(Guo et al.,\n2024a)"
    cleaned_text = re.sub(r'\(([^)]+)\)', lambda m: '(' + m.group(1).replace('\n', ' ') + ')', text)

    citations = []

    for pattern, citation_type in CITATION_PATTERNS:
        for match in re.finditer(pattern, cleaned_text):
            citations.append({
                "text": match.group(0),
                "type": citation_type,
                "char_start": match.start(),
                "char_end": match.end(),
                "citation_content": match.group(1) if (match.lastindex and match.lastindex >= 1) else match.group(0)
            })

    # Sort by position
    citations.sort(key=lambda c: c["char_start"])

    return citations


def identify_paragraphs(blocks: List[Dict], spacing_threshold: float = 1.5) -> List[Dict]:
    """
    Identify paragraph boundaries using vertical spacing between text blocks.

    Args:
        blocks: List of text blocks from PyMuPDF dict output
        spacing_threshold: Multiplier for line height to detect paragraph breaks

    Returns:
        List of paragraph dicts: [{"text": str, "page_num": int, "block_indices": List[int]}]
    """
    if not blocks:
        return []

    paragraphs = []
    current_paragraph = []
    prev_bottom = None
    prev_page = None

    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        if not text:
            continue

        bbox = block.get("bbox")
        page_num = block.get("page_num", 0)

        # Start new paragraph if:
        # 1. Different page
        # 2. Large vertical gap (indicates paragraph break)
        start_new_para = False

        if prev_page is not None and page_num != prev_page:
            start_new_para = True
        elif prev_bottom is not None and bbox:
            # Calculate vertical spacing
            current_top = bbox[1]  # y0
            line_height = bbox[3] - bbox[1]  # y1 - y0
            vertical_gap = current_top - prev_bottom

            # If gap is larger than threshold * line_height, it's a paragraph break
            if line_height > 0 and vertical_gap > (spacing_threshold * line_height):
                start_new_para = True

        if start_new_para and current_paragraph:
            # Finalize current paragraph
            para_text = " ".join([blocks[idx]["text"].strip() for idx in current_paragraph])
            paragraphs.append({
                "text": para_text,
                "page_num": blocks[current_paragraph[0]].get("page_num", 0),
                "block_indices": current_paragraph.copy()
            })
            current_paragraph = []

        # Add block to current paragraph
        current_paragraph.append(i)

        # Update prev_bottom for next iteration
        if bbox:
            prev_bottom = bbox[3]  # y1
        prev_page = page_num

    # Finalize last paragraph
    if current_paragraph:
        para_text = " ".join([blocks[idx]["text"].strip() for idx in current_paragraph])
        paragraphs.append({
            "text": para_text,
            "page_num": blocks[current_paragraph[0]].get("page_num", 0),
            "block_indices": current_paragraph
        })

    return paragraphs


def build_char_offset_map(text: str, blocks: List[Dict]) -> Dict[int, int]:
    """
    Build a mapping from block indices to character offsets in the full text.

    Args:
        text: Full text (concatenated from all blocks)
        blocks: List of text blocks

    Returns:
        Dict mapping block_idx -> char_start in full text
    """
    offset_map = {}
    current_offset = 0

    for i, block in enumerate(blocks):
        block_text = block.get("text", "")
        offset_map[i] = current_offset
        current_offset += len(block_text)

    return offset_map


def assign_paragraphs_to_sections(
    paragraphs: List[Dict],
    sections: List[Dict],
    blocks: List[Dict]
) -> List[Dict]:
    """
    Assign each paragraph to its parent section based on block indices.

    Args:
        paragraphs: List of paragraph dicts with block_indices
        sections: List of section dicts with start_idx and end_idx
        blocks: List of all text blocks

    Returns:
        Enhanced paragraphs with "section" field added
    """
    enhanced_paragraphs = []

    for para in paragraphs:
        # Find which section contains this paragraph's first block
        first_block_idx = para["block_indices"][0]

        section_name = "unknown"
        for section in sections:
            if section["start_idx"] <= first_block_idx <= section["end_idx"]:
                section_name = section["type"]
                break

        enhanced_para = para.copy()
        enhanced_para["section"] = section_name
        enhanced_paragraphs.append(enhanced_para)

    return enhanced_paragraphs


def extract_section_text(section: Dict, blocks: List[Dict]) -> str:
    """
    Extract the full text of a section from blocks.

    Args:
        section: Section dict with start_idx and end_idx
        blocks: List of all text blocks

    Returns:
        Full text of the section (cleaned for PDF artifacts)
    """
    start_idx = section["start_idx"]
    end_idx = section["end_idx"]

    section_blocks = blocks[start_idx:end_idx + 1]
    section_text = " ".join([b.get("text", "").strip() for b in section_blocks if b.get("text", "").strip()])

    # Clean PDF artifacts (newlines in citations, hyphenated words, etc.)
    if section_text:
        # Fix newlines within parentheses (citations)
        section_text = re.sub(r'\(([^)]+)\)', lambda m: '(' + m.group(1).replace('\n', ' ') + ')', section_text)

        # Fix newlines within square brackets
        section_text = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace('\n', ' ') + ']', section_text)

        # Fix hyphenated words
        section_text = re.sub(r'([a-zA-Z])-\s+([a-z])', r'\1\2', section_text)

        # Collapse multiple spaces
        section_text = re.sub(r' +', ' ', section_text)

    return section_text.strip()


def validate_structure(structure: Dict) -> Tuple[bool, List[str]]:
    """
    Validate extracted structure and return warnings/errors.

    Args:
        structure: Structure dict with sections, paragraphs, citations

    Returns:
        (is_valid, warnings) tuple
    """
    warnings = []
    is_valid = True

    sections = structure.get("sections", [])
    paragraphs = structure.get("paragraphs", [])
    citations = structure.get("citations", [])

    # Check for common issues
    if not sections:
        warnings.append("No sections detected - structure extraction may have failed")

    if not paragraphs:
        warnings.append("No paragraphs detected - text may be poorly formatted")

    # Check for expected sections
    section_types = {s["type"] for s in sections}
    expected_sections = {"abstract", "introduction", "conclusion", "references"}
    missing_sections = expected_sections - section_types

    if missing_sections:
        warnings.append(f"Missing common sections: {', '.join(missing_sections)}")

    # Check citation count
    if len(citations) == 0:
        warnings.append("No citations detected - paper may not have standard citation format")

    return is_valid, warnings
