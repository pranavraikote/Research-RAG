from pathlib import Path
import requests
import json
import time
from tqdm import tqdm
from acl_anthology import Anthology

class ACLAnthologyLoader:
    
    def __init__(self, data_dir = "./data/acl", rate_limit_delay = 0.5):
        """
        Initialize ACL Anthology loader.
        
        Args:
            data_dir: Directory to store downloaded papers
            rate_limit_delay: Delay between requests (seconds) to avoid rate limiting
        """

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.base_url = "https://aclanthology.org"
        self.rate_limit_delay = rate_limit_delay
        self.anthology = Anthology.from_repo()
    
    def get_paper_metadata(self, paper_id):
        """
        Getting paper metadata function.
        
        Args:
            paper_id: Paper ID
        
        Returns:
            result: Dictionary with paper metadata
        """
        
        paper = self.anthology.get(paper_id)
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "venue": getattr(paper, 'venue', None),
            "year": getattr(paper, 'year', None),
            "url": f"{self.base_url}/{paper_id}",
            "pdf_url": f"{self.base_url}/{paper_id}.pdf",
            "abstract": getattr(paper, 'abstract', None),
        }
    
    def download_paper(self, paper_id, save_path = None):
        """
        Downloading paper function.
        
        Args:
            paper_id: Paper ID
            save_path: Path to save PDF (optional)
        
        Returns:
            result: Path to downloaded PDF
        """
        
        if save_path is None:
            safe_id = paper_id.replace("/", "_").replace(":", "_")
            save_path = self.data_dir / f"{safe_id}.pdf"
        
        save_path = Path(save_path)
        if save_path.exists():
            return str(save_path)
        
        response = requests.get(f"{self.base_url}/{paper_id}.pdf", timeout=30)
        response.raise_for_status()
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        time.sleep(self.rate_limit_delay)
        return str(save_path)
    
    def list_papers_by_venue(self, venue, year = None, limit = None):
        """
        Listing papers by venue function.
        
        Args:
            venue: Conference venue
            year: Year filter
            limit: Maximum number of papers
        
        Returns:
            result: List of paper IDs
        """
        
        paper_ids = []
        for paper_id, paper in self.anthology.papers.items():
            paper_venue = getattr(paper, 'venue', '').lower()
            paper_year = getattr(paper, 'year', None)
            
            if venue.lower() not in paper_venue:
                continue
            
            if year is not None and paper_year != year:
                continue
            
            paper_ids.append(paper_id)
            
            if limit and len(paper_ids) >= limit:
                break
        
        return paper_ids
    
    def download_papers_bulk(self, paper_ids, save_metadata = True, skip_existing = True):
        """
        Downloading papers in bulk function.
        
        Args:
            paper_ids: List of paper IDs
            save_metadata: Whether to save metadata
            skip_existing: Whether to skip existing PDFs
        
        Returns:
            result: Dictionary with download statistics
        """
        
        downloaded = []
        failed = []
        skipped = []
        
        for paper_id in tqdm(paper_ids, desc = "Downloading papers"):
            try:
                safe_id = paper_id.replace("/", "_").replace(":", "_")
                pdf_path = self.data_dir / f"{safe_id}.pdf"
                
                if skip_existing and pdf_path.exists():
                    skipped.append(paper_id)
                    continue
                
                # Download the paper!
                self.download_paper(paper_id)
                downloaded.append(paper_id)
                
                # Try to get metadata, this will help us a bit :P
                if save_metadata:
                    metadata = self.get_paper_metadata(paper_id)
                    with open(self.metadata_dir / f"{safe_id}.json", "w") as f:
                        json.dump(metadata, f, indent=2)
            
            except Exception as e:
                print(f"\nError downloading {paper_id}: {e}")
                failed.append(paper_id)
        
        return {
            "downloaded": len(downloaded),
            "failed": len(failed),
            "skipped": len(skipped),
            "total": len(paper_ids),
            "downloaded_ids": downloaded,
            "failed_ids": failed,
            "skipped_ids": skipped
        }
    
    def download_by_venue(self, venue, year = None, limit = None, save_metadata = True):
        """
        Downloading papers by venue function.
        
        Args:
            venue: Conference venue
            year: Year filter
            limit: Maximum number of papers
            save_metadata: Whether to save metadata
        
        Returns:
            result: Dictionary with download statistics
        """
        
        print(f"Finding papers from {venue}" + (f" ({year})" if year else ""))
        paper_ids = self.list_papers_by_venue(venue, year=year, limit=limit)
        
        if not paper_ids:
            return {"error": "No papers found"}
        
        print(f"Found {len(paper_ids)} papers. Starting download...")
        
        return self.download_papers_bulk(paper_ids, save_metadata=save_metadata)
    
    def get_downloaded_papers(self):
        """
        Getting downloaded papers function.
        
        Returns:
            result: List of paper IDs
        """
        
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return [f.name.replace(".pdf", "").replace("_", "/") for f in pdf_files]