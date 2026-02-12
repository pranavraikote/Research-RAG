"""
Context expansion for retrieved chunks.

When a chunk is retrieved from a split section, this can optionally
include neighboring chunks for additional context.
"""
from typing import List, Dict, Set


class ContextExpander:
    """
    Expand retrieved chunks with neighboring context from the same section.

    Use cases:
    1. Add previous/next chunk for continuity
    2. Merge multiple retrieved chunks from same section
    3. Retrieve entire section if multiple chunks match
    """

    def __init__(self, all_chunks: List[Dict], all_metadata: List[Dict]):
        """
        Initialize expander with full chunk corpus.

        Args:
            all_chunks: List of all chunk texts
            all_metadata: List of all chunk metadata
        """
        self.all_chunks = all_chunks
        self.all_metadata = all_metadata

        # Build index: chunk_id -> index
        self.chunk_id_to_idx = {
            meta.get("chunk_id"): i
            for i, meta in enumerate(all_metadata)
        }

    def expand_with_neighbors(
        self,
        retrieved_chunks: List[Dict],
        window_size: int = 1,
        max_chunks_per_section: int = 5
    ) -> List[Dict]:
        """
        Add neighboring chunks from the same section.

        Strategy:
        - For each retrieved chunk, add N chunks before and after
        - Only add chunks from the SAME section (same section_type + paper)
        - Remove duplicates
        - Sort by original chunk order

        Args:
            retrieved_chunks: List of retrieved chunk dicts with metadata
            window_size: Number of chunks to add before/after (default: 1)
            max_chunks_per_section: Max chunks to return per section

        Returns:
            Expanded list of chunks with added context
        """
        expanded_indices = set()

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            chunk_id = metadata.get("chunk_id", "")

            if chunk_id not in self.chunk_id_to_idx:
                continue

            base_idx = self.chunk_id_to_idx[chunk_id]
            expanded_indices.add(base_idx)

            # Get section and paper info
            section_type = metadata.get("section_type", "")
            chunk_id_str = str(chunk_id)  # Convert to string for operations
            paper_id = chunk_id_str.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id_str else ""

            # Add neighbors within same section
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = base_idx + offset

                if 0 <= neighbor_idx < len(self.all_metadata):
                    neighbor_meta = self.all_metadata[neighbor_idx]
                    neighbor_section = neighbor_meta.get("section_type", "")
                    neighbor_id = neighbor_meta.get("chunk_id", "")
                    neighbor_id_str = str(neighbor_id)  # Convert to string for operations
                    neighbor_paper = neighbor_id_str.rsplit("_chunk_", 1)[0] if "_chunk_" in neighbor_id_str else ""

                    # Only add if same section AND same paper
                    if neighbor_section == section_type and neighbor_paper == paper_id:
                        expanded_indices.add(neighbor_idx)

        # Convert indices to chunks with scores
        # Preserve scores from original retrieved chunks
        original_scores = {}
        original_indices = set()
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_id")
            if chunk_id and chunk_id in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[chunk_id]
                original_scores[idx] = chunk.get("score", 0.0)
                original_indices.add(idx)

        # Build expanded chunks in DOCUMENT ORDER (reading order)
        # Original chunks keep scores, neighbors get 0.0
        expanded_chunks = []
        for idx in sorted(expanded_indices):  # Sort by document position
            expanded_chunks.append({
                "text": self.all_chunks[idx],
                "metadata": self.all_metadata[idx],
                "score": original_scores.get(idx, 0.0),  # Original score or 0.0 for neighbors
                "is_expanded": idx not in original_indices  # Mark context vs retrieved
            })

        # Return in document order for natural reading flow
        return expanded_chunks

    def merge_section_chunks(
        self,
        retrieved_chunks: List[Dict],
        merge_threshold: int = 2
    ) -> List[Dict]:
        """
        If multiple chunks from same section are retrieved, merge them.

        Strategy:
        - Group chunks by (paper_id, section_type)
        - If a section has >= merge_threshold chunks, merge them all
        - Otherwise keep individual chunks

        Args:
            retrieved_chunks: List of retrieved chunk dicts
            merge_threshold: Min chunks to trigger merge (default: 2)

        Returns:
            List of chunks, with same-section chunks merged
        """
        # Group by section
        section_groups = {}

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            chunk_id = metadata.get("chunk_id", "")
            section_type = metadata.get("section_type", "")
            chunk_id_str = str(chunk_id)  # Convert to string for operations
            paper_id = chunk_id_str.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id_str else ""

            key = (paper_id, section_type)
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(chunk)

        # Merge or keep individual
        result = []

        for (paper_id, section_type), chunks in section_groups.items():
            if len(chunks) >= merge_threshold:
                # Merge chunks from this section
                merged_text = "\n\n".join([c["text"] for c in chunks])
                merged_metadata = chunks[0]["metadata"].copy()
                merged_metadata["chunk_id"] = f"{paper_id}_merged_{section_type}"
                merged_metadata["merged_from"] = [c["metadata"]["chunk_id"] for c in chunks]
                merged_metadata["merged_chunk_count"] = len(chunks)

                # Use max score from merged chunks
                merged_score = max([c.get("score", 0.0) for c in chunks])

                result.append({
                    "text": merged_text,
                    "metadata": merged_metadata,
                    "score": merged_score  # Preserve score
                })
            else:
                # Keep individual chunks (already have scores)
                result.extend(chunks)

        return result

    def get_full_section(
        self,
        chunk: Dict
    ) -> List[Dict]:
        """
        Get ALL chunks from the same section as the given chunk.

        Use this when you want complete section context.

        Args:
            chunk: A single chunk dict with metadata

        Returns:
            List of all chunks from that section, in order
        """
        metadata = chunk.get("metadata", {})
        chunk_id = metadata.get("chunk_id", "")
        section_type = metadata.get("section_type", "")
        paper_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else ""

        section_chunks = []

        for i, meta in enumerate(self.all_metadata):
            meta_id = meta.get("chunk_id", "")
            meta_section = meta.get("section_type", "")
            meta_paper = meta_id.rsplit("_chunk_", 1)[0] if "_chunk_" in meta_id else ""

            if meta_section == section_type and meta_paper == paper_id:
                section_chunks.append({
                    "text": self.all_chunks[i],
                    "metadata": meta
                })

        return section_chunks


# Example usage
if __name__ == "__main__":
    """
    Example: How to use context expansion with retrieval
    """

    # Mock data for demonstration
    chunks = ["chunk1", "chunk2", "chunk3"]
    metadata = [
        {"chunk_id": "paper1_chunk_0", "section_type": "methods"},
        {"chunk_id": "paper1_chunk_1", "section_type": "methods"},
        {"chunk_id": "paper1_chunk_2", "section_type": "results"},
    ]

    retrieved = [
        {"text": "chunk2", "metadata": metadata[1]}
    ]

    expander = ContextExpander(chunks, metadata)

    # Add 1 chunk before/after
    expanded = expander.expand_with_neighbors(retrieved, window_size=1)
    print(f"Expanded from {len(retrieved)} to {len(expanded)} chunks")

    # Get full section
    full_section = expander.get_full_section(retrieved[0])
    print(f"Full section has {len(full_section)} chunks")
