"""PDF document loader."""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load PDF documents."""
    
    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images
        
        # Try to import PDF libraries
        self.pdf_library = self._get_pdf_library()
    
    def _get_pdf_library(self):
        """Get available PDF library."""
        try:
            import PyPDF2
            return "pypdf2"
        except ImportError:
            pass
        
        try:
            import pdfplumber
            return "pdfplumber"
        except ImportError:
            pass
        
        logger.warning("No PDF library found. Install PyPDF2 or pdfplumber.")
        return None
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a PDF file."""
        if not self.pdf_library:
            raise ImportError("No PDF library available. Install PyPDF2 or pdfplumber.")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        metadata = {
            "source": str(file_path),
            "file_type": "pdf",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        }
        
        if self.pdf_library == "pypdf2":
            content = self._load_with_pypdf2(file_path)
        elif self.pdf_library == "pdfplumber":
            content = self._load_with_pdfplumber(file_path)
        else:
            raise RuntimeError("No PDF library available")
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _load_with_pypdf2(self, file_path: Path) -> str:
        """Load PDF using PyPDF2."""
        import PyPDF2
        
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if key.startswith('/'):
                        key = key[1:]
                    metadata[key] = str(value)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
        
        return "\n\n".join(text_parts)
    
    def _load_with_pdfplumber(self, file_path: Path) -> str:
        """Load PDF using pdfplumber."""
        import pdfplumber
        
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            # Extract metadata
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    if value:
                        metadata[key] = str(value)
            
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    for table in tables:
                        table_text = self._format_table(table)
                        if table_text:
                            text_parts.append(f"\n[Table on page {page_num + 1}]\n{table_text}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract content from page {page_num + 1}: {e}")
        
        return "\n\n".join(text_parts)
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format table data as text."""
        if not table:
            return ""
        
        # Simple text representation of table
        lines = []
        for row in table:
            line = " | ".join(str(cell or "") for cell in row)
            lines.append(line)
        
        return "\n".join(lines)
    
    def load_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple PDF files."""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load PDF {file_path}: {e}")
        
        return documents