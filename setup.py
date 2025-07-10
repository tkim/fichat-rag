"""Setup script for fichat-rag."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fichat-rag",
    version="0.1.0",
    author="FI-Chat Team",
    description="A production-ready RAG framework extracted from FI-Chat",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fichat-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0",
        "tiktoken>=0.5.0",
        "psycopg2-binary>=2.9.0",
        "pgvector>=0.2.0",
        "asyncpg>=0.29.0",
        "nltk>=3.8.0",
        "chardet>=5.0.0",
    ],
    extras_require={
        "pdf": ["PyPDF2>=3.0.0", "pdfplumber>=0.10.0"],
        "web": ["firecrawl-py>=0.0.14"],
        "chromadb": ["chromadb>=0.4.0"],
        "qdrant": ["qdrant-client>=1.7.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "PyPDF2>=3.0.0",
            "pdfplumber>=0.10.0",
            "firecrawl-py>=0.0.14",
            "chromadb>=0.4.0",
            "qdrant-client>=1.7.0",
            "langdetect>=1.0.9",
            "pyyaml>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fichat-rag=src.cli:main",
        ],
    },
)