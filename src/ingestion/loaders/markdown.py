"""Markdown document loader."""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """Load Markdown documents."""
    
    def __init__(
        self,
        remove_images: bool = False,
        remove_links: bool = False,
        extract_metadata: bool = True
    ):
        """
        Initialize Markdown loader.
        
        Args:
            remove_images: Whether to remove image references
            remove_links: Whether to remove links (keep text only)
            extract_metadata: Whether to extract frontmatter metadata
        """
        self.remove_images = remove_images
        self.remove_links = remove_links
        self.extract_metadata = extract_metadata
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a Markdown file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "source": str(file_path),
            "file_type": "markdown",
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size
        }
        
        # Extract frontmatter if present
        if self.extract_metadata and content.startswith('---'):
            try:
                _, frontmatter, content = content.split('---', 2)
                # Parse YAML frontmatter
                import yaml
                fm_data = yaml.safe_load(frontmatter)
                if isinstance(fm_data, dict):
                    metadata.update(fm_data)
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter in {file_path}: {e}")
        
        # Process content
        if self.remove_images:
            # Remove image syntax ![alt](url)
            content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', content)
        
        if self.remove_links:
            # Convert links [text](url) to just text
            content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Extract title from first header if not in metadata
        if 'title' not in metadata:
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
        
        # Count headers and links for metadata
        metadata['header_count'] = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        metadata['link_count'] = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content))
        
        return {
            "content": content.strip(),
            "metadata": metadata
        }
    
    def load_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple Markdown files."""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load Markdown file {file_path}: {e}")
        
        return documents