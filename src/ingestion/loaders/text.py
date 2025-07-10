"""Text file loader."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import chardet
import logging

logger = logging.getLogger(__name__)


class TextLoader:
    """Load text documents."""
    
    def __init__(self, encoding: Optional[str] = None, errors: str = 'ignore'):
        self.encoding = encoding
        self.errors = errors
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a text file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        # Detect encoding if not specified
        encoding = self.encoding
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        # Read file
        try:
            with open(file_path, 'r', encoding=encoding, errors=self.errors) as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode {file_path} with {encoding}, trying utf-8")
            with open(file_path, 'r', encoding='utf-8', errors=self.errors) as f:
                content = f.read()
        
        metadata = {
            "source": str(file_path),
            "file_type": file_path.suffix.lower()[1:] if file_path.suffix else "txt",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "encoding": encoding
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                if confidence < 0.7:
                    logger.warning(
                        f"Low confidence ({confidence}) for encoding detection: {encoding}. "
                        "Defaulting to utf-8"
                    )
                    return 'utf-8'
                
                return encoding or 'utf-8'
        except Exception as e:
            logger.warning(f"Failed to detect encoding: {e}. Using utf-8")
            return 'utf-8'
    
    def load_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple text files."""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load text file {file_path}: {e}")
        
        return documents