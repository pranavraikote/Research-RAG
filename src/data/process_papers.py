import sys
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import PDFLoader
from src.chunking.basic import BasicChunker
from src.embeddings import EmbeddingGenerator

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.semantic import SemanticRetriever


class PaperProcessor:
    
    def __init__(self, data_dir = None, chunk_size = 1000, chunk_overlap = 200, index_path = None, metric = "IP"):
        """
        Initialize paper processor.
        
        Args:
            data_dir: Directory with PDF files
            chunk_size: Chunk size in characters
            chunk_overlap: Chunk overlap in characters
            index_path: Path to FAISS index
            metric: Distance metric ("IP" or "L2")
        """
        
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "acl"
        
        self.data_dir = Path(data_dir)
        self.pdf_loader = PDFLoader()
        self.chunker = BasicChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = SemanticRetriever(index_path=index_path, metric=metric)
        self.bm25_retriever = BM25Retriever()
    
    def load_paper_metadata(self, paper_id):
        """
        Loading paper metadata function.
        
        Args:
            paper_id: Paper ID
        
        Returns:
            result: Dictionary with paper metadata
        """
        
        metadata_path = self.data_dir / "metadata" / f"{paper_id.replace('/', '_').replace(':', '_')}.json"
        
        # Checking for metadata file
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"paper_id": paper_id}
    
    def process_paper(self, pdf_path):
        """
        Processing a single paper function.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            result: Dictionary with processing results
        """
        
        paper_id = pdf_path.stem
        paper_metadata = self.load_paper_metadata(paper_id)
        
        # Start processing and load the pdf
        try:
            pdf_data = self.pdf_loader.load_pdf(str(pdf_path))
        except Exception as e:
            return {
                "paper_id": paper_id,
                "status": "failed",
                "reason": f"PDF loading error: {str(e)[:100]}"
            }
        
        text = pdf_data["text"]
        
        # Some cases to handle :)
        if not text or len(text.strip()) < 100:
            return {
                "paper_id": paper_id,
                "status": "failed",
                "reason": "No text extracted or text too short"
            }
        
        # Trying to get title from multiple sources
        title = (
            paper_metadata.get("title") or 
            pdf_data["metadata"].get("title") or 
            ""
        )
        
        # If title is still empty or looks like a header, extracting from text
        if not title or title.strip() == "" or "Proceedings of" in title or "Annual Meeting" in title:
            title = self.chunker._extract_title_from_text(text)
        
        # Preparing metdata payload :)
        chunk_metadata = {
            "paper_id": paper_id,
            "paper_title": title,
            "conference": paper_metadata.get("venue", "ACL"),
            "year": paper_metadata.get("year", 2025),
        }
        
        chunks = self.chunker.chunk(text, metadata = chunk_metadata)
        
        if not chunks:
            return {
                "paper_id": paper_id,
                "status": "failed",
                "reason": "No chunks created"
            }
        
        # Generating embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_gen.embed_texts(chunk_texts)
        
        return {
            "paper_id": paper_id,
            "status": "success",
            "num_chunks": len(chunks),
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": paper_metadata
        }
    
    def process_all_papers(self, limit = None):
        """
        Processing all papers in the data directory function.
        
        Args:
            limit: Maximum number of papers to process
        
        Returns:
            result: Dictionary with processing statistics
        """
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        print(f"Processing {len(pdf_files)} papers...")
        
        processed = []
        failed = []
        all_chunks_list = []
        all_embeddings_list = []
        all_chunk_texts = []
        
        # Bulk processing papers
        for pdf_path in tqdm(pdf_files, desc = "Processing papers"):
            result = self.process_paper(pdf_path)
            
            if result["status"] == "success":
                processed.append(result["paper_id"])
                all_chunks_list.extend(result["chunks"])
                all_embeddings_list.append(result["embeddings"])
                all_chunk_texts.extend([chunk["text"] for chunk in result["chunks"]])
            else:
                failed.append({
                    "paper_id": result["paper_id"],
                    "reason": result.get("reason", "Unknown")
                })
        
        # Checking existing embeddings and adding the freshly processed chunks too
        if all_embeddings_list:
            all_embeddings = np.vstack(all_embeddings_list)
            
            print(f"\nAdding {len(all_chunks_list)} chunks to vector stores...")
            
            chunk_metadata_list = [chunk["metadata"] for chunk in all_chunks_list]
            self.vector_store.add_chunks(
                embeddings=all_embeddings,
                chunks=all_chunk_texts,
                metadata=chunk_metadata_list
            )
            
            self.bm25_retriever.add_chunks(
                chunks=all_chunk_texts,
                metadata=chunk_metadata_list
            )
            
            print(f"Added {len(all_chunks_list)} chunks to FAISS index")
            print(f"Added {len(all_chunks_list)} chunks to BM25 index")
        
        self._all_chunks_list = all_chunks_list
        
        return {
            "processed": len(processed),
            "failed": len(failed),
            "total_chunks": len(all_chunks_list),
            "processed_papers": processed,
            "failed_papers": failed
        }
    
    def save_index(self, index_path):
        """
        Saving FAISS index to disk function.
        
        Args:
            index_path: Path to save the index
        """
        self.vector_store.save_index(index_path)
        print(f"Saved FAISS index to {index_path}")
    
    def save_chunks(self, chunks_path):
        """
        Saving chunks to JSON file function.
        
        Args:
            chunks_path: Path to save chunks JSON file
        """
        if hasattr(self, '_all_chunks_list') and self._all_chunks_list:
            self.chunker.save_chunks(self._all_chunks_list, chunks_path)
            print(f"Saved chunks to {chunks_path}")
        else:
            print("No chunks to save. Process papers first.")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    project_root = Path(__file__).parent.parent.parent
    default_data_dir = str(project_root / "data" / "acl")
    default_index_path = str(project_root / "artifacts" / "faiss_index")
    
    parser.add_argument("--data-dir", default=default_data_dir, help="Directory with PDFs")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--limit", type=int, help="Limit number of papers to process")
    parser.add_argument("--index-path", default=default_index_path, help="Path to save FAISS index")
    parser.add_argument("--metric", default="IP", choices=['IP', 'L2'], help="Distance metric (IP=cosine similarity, L2=Euclidean distance)")
    
    args = parser.parse_args()
    
    # Initializing the mighty Processor!
    processor = PaperProcessor(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        index_path=args.index_path,
        metric=args.metric
    )
    
    stats = processor.process_all_papers(limit=args.limit)

    print("Processing Summary:")
    print(f"Processed: {stats['processed']} papers")
    print(f"Failed: {stats['failed']} papers")
    print(f"Total chunks: {stats['total_chunks']}")
    
    processor.save_index(args.index_path)
    
    chunks_path = str(Path(args.index_path).parent / "chunks.json")
    processor.save_chunks(chunks_path)
    
    print("\nProcessing complete!")