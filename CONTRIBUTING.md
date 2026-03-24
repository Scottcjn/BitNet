# Contributing to BitNet

Thank you for your interest in contributing to BitNet! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Recognition](#recognition)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. By participating, you are expected to:

- Be respectful and inclusive in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback and collaboration
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BitNet.git
   cd BitNet
   ```
3. **Set up the development environment** (see [Development Setup](#development-setup))
4. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Check the [existing issues](../../issues) to avoid duplicates
2. Update to the latest version to see if the issue is already resolved
3. Collect information about the bug (steps to reproduce, expected vs actual behavior)

When submitting a bug report, include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Any error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- Clear description of the proposed feature
- Use cases and benefits
- Possible implementation approach (if you have ideas)
- Willingness to contribute the implementation

### Pull Requests

1. Fork the repository and create your branch from `main`
2. Make your changes following our [coding standards](#coding-standards)
3. Add or update tests as necessary
4. Update documentation if needed
5. Ensure all tests pass
6. Submit a pull request with a clear description

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Scottcjn/BitNet.git
cd BitNet

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Keep functions focused and concise
- Add docstrings to all public functions and classes

Example:
```python
def process_bitnet_layer(input_tensor, bit_width=8):
    """
    Process input through a BitNet layer with quantized weights.
    
    Args:
        input_tensor: Input tensor of shape (batch_size, features)
        bit_width: Number of bits for quantization (default: 8)
    
    Returns:
        Processed tensor after BitNet transformation
    """
    # Implementation here
    pass
```

### Code Organization

- Keep related functionality in the same module
- Separate concerns (data processing, model logic, utilities)
- Use type hints where appropriate
- Avoid circular imports

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Include code examples for complex features
- Keep comments current with code changes

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=bitnet --cov-report=html

# Run specific test file
pytest tests/test_specific.py
```

### Writing Tests

- Add tests for new functionality
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests independent and isolated

Example:
```python
def test_bitnet_quantization_preserves_shape():
    """Test that quantization maintains tensor dimensions."""
    input_tensor = torch.randn(32, 128)
    quantized = quantize_weights(input_tensor, bit_width=8)
    assert quantized.shape == input_tensor.shape
```

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests where appropriate

Examples:
```
Add INT8 quantization support for linear layers

Implement 8-bit quantization for BitNet linear layers with
learnable scaling factors. Includes tests and documentation.

Fixes #123
```

### Pull Request Process

1. Update your fork to the latest `main` branch
2. Push your branch to your fork
3. Open a Pull Request against the main repository
4. Fill out the PR template with:
   - Description of changes
   - Motivation and context
   - Testing performed
   - Screenshots (if applicable)
5. Wait for review and address feedback
6. Once approved, your PR will be merged

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] PR description explains the changes

## Recognition

Contributors will be recognized in our README.md file and release notes. Significant contributions may be invited to become maintainers.

Thank you for contributing to BitNet!
