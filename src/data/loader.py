import fitz
from pathlib import Path

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
    
    def _clean_pdf_text(self, text):
        """
        Cleaning text extracted from PDF function.
        
        Args:
            text: Raw text from PDF
        
        Returns:
            result: Cleaned text
        """
        
        import re
        
        if not text:
            return ""
        
        # Replacing multiple newlines with single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fixing word breaks: if a line ends with a hyphen and next line starts with lowercase,
        # it's likely a word split across lines
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Checking if this line ends with hyphen and next line starts with lowercase
            if (i < len(lines) - 1 and 
                line and line[-1] == '-' and 
                lines[i+1].strip() and lines[i+1].strip()[0].islower()):
                # Merging: remove hyphen and join
                next_line = lines[i+1].strip()
                merged = line[:-1] + next_line
                cleaned_lines.append(merged)
                i += 2
            else:
                cleaned_lines.append(line)
                i += 1
        
        text = '\n'.join(cleaned_lines)
        
        # Removing standalone numbers/artifacts
        text = re.sub(r'^\d+\.\s*$', '', text, flags=re.MULTILINE)

        # Cleaning up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()