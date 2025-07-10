"""Document processors for cleaning and preparing text."""

import re
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and clean documents before chunking."""
    
    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False
    ):
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
    
    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Process text with configured cleaning steps."""
        if not text:
            return ""
        
        # Store original length
        original_length = len(text)
        
        # Apply processing steps
        if self.remove_urls:
            text = self._remove_urls(text)
        
        if self.remove_emails:
            text = self._remove_emails(text)
        
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)
        
        # Log processing stats
        if metadata:
            metadata['processing_stats'] = {
                'original_length': original_length,
                'processed_length': len(text),
                'reduction_percent': round((1 - len(text) / original_length) * 100, 2)
            }
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, ' ', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, ' ', text)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        
        # Normalize to NFKD form
        text = unicodedata.normalize('NFKD', text)
        
        # Replace common unicode characters
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u2026': '...',  # Horizontal ellipsis
            '\u00a0': ' ',  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        return re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-\:\;\'\"]', ' ', text)
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class MetadataExtractor:
    """Extract metadata from documents."""
    
    def extract(self, text: str, existing_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract metadata from text."""
        metadata = existing_metadata or {}
        
        # Extract basic statistics
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['line_count'] = len(text.splitlines())
        
        # Extract potential title (first non-empty line)
        lines = text.strip().splitlines()
        if lines:
            potential_title = lines[0].strip()
            if len(potential_title) < 200:  # Reasonable title length
                metadata['extracted_title'] = potential_title
        
        # Detect language (simple heuristic)
        metadata['language'] = self._detect_language(text)
        
        # Extract dates
        dates = self._extract_dates(text)
        if dates:
            metadata['extracted_dates'] = dates
        
        return metadata
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        # This is a very basic implementation
        # For production, use langdetect or similar library
        
        # Check for common English words
        english_words = {'the', 'is', 'are', 'was', 'were', 'have', 'has', 'been', 'and', 'or', 'but'}
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if f' {word} ' in text_lower)
        
        if english_count > 5:
            return 'en'
        
        return 'unknown'
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',     # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates


class TableExtractor:
    """Extract and process tables from documents."""
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from text (basic implementation)."""
        tables = []
        
        # Look for table-like structures (lines with consistent delimiters)
        lines = text.splitlines()
        current_table = []
        
        for i, line in enumerate(lines):
            # Check if line looks like a table row
            if self._is_table_row(line):
                current_table.append(line)
            else:
                # End of table
                if len(current_table) > 2:  # At least header + 1 data row
                    table_data = self._parse_table(current_table)
                    if table_data:
                        tables.append({
                            'start_line': i - len(current_table),
                            'end_line': i - 1,
                            'data': table_data
                        })
                current_table = []
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Simple heuristic: contains multiple | or tab characters
        return line.count('|') >= 2 or line.count('\t') >= 2
    
    def _parse_table(self, lines: List[str]) -> Optional[Dict[str, List[Any]]]:
        """Parse table lines into structured data."""
        if not lines:
            return None
        
        # Determine delimiter
        delimiter = '|' if '|' in lines[0] else '\t'
        
        # Parse header
        header = [cell.strip() for cell in lines[0].split(delimiter)]
        
        # Parse data rows
        data = {col: [] for col in header}
        
        for line in lines[1:]:
            cells = [cell.strip() for cell in line.split(delimiter)]
            for i, col in enumerate(header):
                if i < len(cells):
                    data[col].append(cells[i])
        
        return data if any(data.values()) else None