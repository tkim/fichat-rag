# Contributing to FI-Chat RAG

We welcome contributions to the FI-Chat RAG framework! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/fichat-rag.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Submit a pull request

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Run Tests

```bash
pytest tests/
```

### 3. Code Formatting

We use Black for code formatting:

```bash
black src/ tests/
```

### 4. Linting

Run flake8 for linting:

```bash
flake8 src/ tests/
```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and small
- Write descriptive variable names

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Include both unit and integration tests

## Documentation

- Update documentation for new features
- Include docstrings in your code
- Add examples for new functionality
- Update README if needed

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add a descriptive PR title and description
4. Link any related issues
5. Wait for code review

## Areas for Contribution

- New vector store implementations
- Additional LLM providers
- Better chunking strategies
- Performance optimizations
- Documentation improvements
- Bug fixes
- New examples

## Questions?

Feel free to open an issue for any questions about contributing.