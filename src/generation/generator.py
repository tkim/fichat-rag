"""Response generation for RAG."""

from typing import List, Optional, Iterator
import logging

from ..core.llm import BaseLLM
from ..storage.base import Document
from .prompts import PromptTemplate, DEFAULT_RAG_PROMPT

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using retrieved documents."""
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_context_length: int = 3000,
        include_metadata: bool = False
    ):
        self.llm = llm
        self.prompt_template = PromptTemplate(prompt_template or DEFAULT_RAG_PROMPT)
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> str:
        """Generate response for query using documents."""
        # Format context from documents
        context = self._format_context(documents)
        
        # Create prompt
        prompt = self.prompt_template.format(
            query=query,
            context=context
        )
        
        # Add system prompt if using chat interface
        if self.system_prompt and hasattr(self.llm, 'chat'):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, **kwargs)
        else:
            response = self.llm.generate(prompt, **kwargs)
        
        return response.content
    
    def stream_generate(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> Iterator[str]:
        """Stream generate response."""
        # Format context from documents
        context = self._format_context(documents)
        
        # Create prompt
        prompt = self.prompt_template.format(
            query=query,
            context=context
        )
        
        # Stream response
        for chunk in self.llm.stream_generate(prompt, **kwargs):
            yield chunk
    
    async def agenerate(
        self,
        query: str,
        documents: List[Document],
        **kwargs
    ) -> str:
        """Async generate response."""
        # Format context from documents
        context = self._format_context(documents)
        
        # Create prompt
        prompt = self.prompt_template.format(
            query=query,
            context=context
        )
        
        # Generate response
        response = await self.llm.agenerate(prompt, **kwargs)
        return response.content
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Format document
            if self.include_metadata:
                doc_text = f"Document {i+1}:\n"
                
                # Add relevant metadata
                if 'source' in doc.metadata:
                    doc_text += f"Source: {doc.metadata['source']}\n"
                if 'page' in doc.metadata:
                    doc_text += f"Page: {doc.metadata['page']}\n"
                
                doc_text += f"\n{doc.content}\n"
            else:
                doc_text = f"Document {i+1}:\n{doc.content}\n"
            
            # Check length
            if current_length + len(doc_text) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - current_length
                if remaining > 100:  # Only add if substantial
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n---\n".join(context_parts)