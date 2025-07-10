"""Prompt templates for RAG."""

from typing import Dict, Any


class PromptTemplate:
    """Simple prompt template class."""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)


# Default RAG prompt
DEFAULT_RAG_PROMPT = """Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find information about this in the provided documents."

Context:
{context}

Question: {query}

Answer:"""


# Alternative prompts for different use cases
ACADEMIC_RAG_PROMPT = """You are an academic assistant. Answer the following question based solely on the provided academic sources. Cite specific documents when making claims.

Sources:
{context}

Question: {query}

Academic Response:"""


CONVERSATIONAL_RAG_PROMPT = """You are a helpful assistant. Use the following information to answer the user's question in a natural, conversational way.

Information:
{context}

User: {query}