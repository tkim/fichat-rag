"""JSON document loader."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class JSONLoader:
    """Load JSON documents."""
    
    def __init__(
        self,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        flatten: bool = True
    ):
        """
        Initialize JSON loader.
        
        Args:
            text_fields: Fields to extract as content (if None, converts entire JSON to string)
            metadata_fields: Fields to include in metadata
            flatten: Whether to flatten nested JSON structures
        """
        self.text_fields = text_fields
        self.metadata_fields = metadata_fields
        self.flatten = flatten
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
        
        # Extract content
        if self.text_fields:
            content_parts = []
            for field in self.text_fields:
                value = self._get_nested_value(data, field)
                if value:
                    content_parts.append(f"{field}: {value}")
            content = "\n".join(content_parts)
        else:
            # Convert entire JSON to formatted string
            content = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Extract metadata
        metadata = {
            "source": str(file_path),
            "file_type": "json",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        }
        
        if self.metadata_fields:
            for field in self.metadata_fields:
                value = self._get_nested_value(data, field)
                if value is not None:
                    metadata[field] = value
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def _get_nested_value(self, data: Dict, field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = field_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def load_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple JSON files."""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load JSON file {file_path}: {e}")
        
        return documents