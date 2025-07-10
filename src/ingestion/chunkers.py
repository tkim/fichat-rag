"""Text chunking strategies for document processing."""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class BaseChunker(ABC):
    """Base class for text chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Chunk text into smaller pieces.
        
        Returns:
            List of tuples (chunk_text, start_pos, end_pos)
        """
        pass


class TextChunker(BaseChunker):
    """Simple fixed-size text chunker based on character count."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by character count."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep.rstrip())
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start, end))
            
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks


class SemanticChunker(BaseChunker):
    """Token-based chunker using tiktoken for accurate token counting."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding)
    
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text based on token count."""
        # Encode the entire text
        tokens = self.encoder.encode(text)
        
        if not tokens:
            return []
        
        chunks = []
        start_token = 0
        
        while start_token < len(tokens):
            # Get chunk tokens
            end_token = min(start_token + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            
            # Decode to get text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Find character positions
            # This is approximate due to token/character mismatch
            if chunks:
                start_char = chunks[-1][2] - self.chunk_overlap * 4  # Approximate
            else:
                start_char = 0
            
            end_char = start_char + len(chunk_text)
            
            chunks.append((chunk_text, start_char, end_char))
            
            # Move to next chunk with overlap
            start_token = end_token - self.chunk_overlap
            if start_token <= 0:
                start_token = end_token
        
        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-based chunker that respects sentence boundaries."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 1,  # Number of sentences
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text by sentences."""
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, chunk_start, chunk_end))
                
                # Prepare next chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_sentences = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) + 1 for s in overlap_sentences)
                    chunk_start = chunk_end - current_size
                else:
                    current_chunk = []
                    current_size = 0
                    chunk_start = chunk_end + 1
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add final chunk
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunk_end = chunk_start + len(chunk_text)
            chunks.append((chunk_text, chunk_start, chunk_end))
        
        return chunks


class HybridChunker(BaseChunker):
    """
    Hybrid chunker that combines semantic and structural chunking.
    Respects document structure while maintaining token limits.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding: str = "cl100k_base",
        respect_sections: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding)
        self.respect_sections = respect_sections
    
    def chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text using hybrid approach."""
        if self.respect_sections:
            # Split by sections (headers, paragraphs, etc.)
            sections = self._split_sections(text)
            
            chunks = []
            for section_text, section_start, section_end in sections:
                # Chunk each section independently
                section_chunks = self._chunk_section(section_text, section_start)
                chunks.extend(section_chunks)
            
            return chunks
        else:
            # Fall back to semantic chunking
            return SemanticChunker(
                self.chunk_size,
                self.chunk_overlap,
                self.encoder.name
            ).chunk(text)
    
    def _split_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into logical sections."""
        # Pattern for section headers
        header_pattern = r'\n\n(?=[A-Z#*])'
        
        sections = []
        current_pos = 0
        
        # Split by double newlines followed by capital letter or markdown header
        parts = re.split(header_pattern, text)
        
        for part in parts:
            if part.strip():
                start = current_pos
                end = start + len(part)
                sections.append((part, start, end))
                current_pos = end
        
        return sections if sections else [(text, 0, len(text))]
    
    def _chunk_section(
        self,
        section_text: str,
        section_start: int
    ) -> List[Tuple[str, int, int]]:
        """Chunk a single section respecting token limits."""
        tokens = self.encoder.encode(section_text)
        
        if len(tokens) <= self.chunk_size:
            # Section fits in one chunk
            return [(section_text, section_start, section_start + len(section_text))]
        
        # Need to split section
        chunks = []
        start_token = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            
            # Try to break at sentence boundary
            if end_token < len(tokens):
                chunk_text = self.encoder.decode(chunk_tokens)
                
                # Look for sentence end
                for sep in ['. ', '! ', '? ', '\n']:
                    pos = chunk_text.rfind(sep)
                    if pos > len(chunk_text) * 0.5:  # At least halfway through
                        # Re-encode to get exact token count
                        truncated_text = chunk_text[:pos + len(sep.rstrip())]
                        chunk_tokens = self.encoder.encode(truncated_text)
                        end_token = start_token + len(chunk_tokens)
                        break
            
            chunk_text = self.encoder.decode(chunk_tokens)
            char_offset = len(self.encoder.decode(tokens[:start_token]))
            
            chunks.append((
                chunk_text,
                section_start + char_offset,
                section_start + char_offset + len(chunk_text)
            ))
            
            # Move to next chunk with overlap
            start_token = end_token - self.chunk_overlap
            if start_token <= 0:
                start_token = end_token
        
        return chunks