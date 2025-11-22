# Contributing Guidelines

Thank you for your interest in contributing to the ML-Powered Product Recommendation System! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. Read the [Getting Started Guide](GETTING_STARTED.md)
2. Set up your development environment
3. Familiarized yourself with the codebase structure
4. Read the relevant documentation

### Finding Issues to Work On

- Check the [Issues](https://github.com/yourusername/machinelearningAWS-project/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/machinelearningAWS-project.git
cd machinelearningAWS-project

# Add upstream remote
git remote add upstream https://github.com/original/machinelearningAWS-project.git
```

### 2. Create a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

**Branch Naming Convention**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 3. Make Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add user preference filtering to recommendations"
```

**Commit Message Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example**:
```
feat: add diversity penalty to recommendation ranking

Implemented a diversity penalty in the ranking algorithm to ensure
recommendations include products from multiple categories.

- Added diversity_score calculation
- Updated ranking logic to incorporate diversity
- Added unit tests for diversity penalty

Closes #123
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

**Line Length**: Maximum 100 characters (not 79)

**Imports**:
```python
# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from src.models.collaborative_filtering import NCFModel
from src.utils.logging_utils import get_logger
```

**Type Hints**:
```python
def get_recommendations(
    user_id: str,
    num_recommendations: int = 10,
    exclude_purchased: bool = True
) -> List[Dict[str, Any]]:
    """Get personalized recommendations for a user."""
    pass
```

**Docstrings** (Google Style):
```python
def train_model(data: pd.DataFrame, config: Dict[str, Any]) -> torch.nn.Module:
    """Train the recommendation model.
    
    Args:
        data: Training data with user-item interactions
        config: Configuration dictionary with hyperparameters
            - learning_rate: Learning rate for optimizer
            - batch_size: Batch size for training
            - num_epochs: Number of training epochs
    
    Returns:
        Trained PyTorch model
    
    Raises:
        ValueError: If data is empty or invalid
        RuntimeError: If training fails
    
    Example:
        >>> config = {'learning_rate': 0.001, 'batch_size': 256}
        >>> model = train_model(train_data, config)
    """
    pass
```

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

### Linting

We use **flake8** and **pylint**:

```bash
# Run flake8
flake8 src/ tests/

# Run pylint
pylint src/
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Run mypy
mypy src/
```

## Testing Guidelines

### Writing Tests

**Test Structure**:
```python
import pytest
from src.models.collaborative_filtering import NCFModel

class TestNCFModel:
    """Test suite for NCF model."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample training data."""
        return {
            'user_ids': [1, 2, 3],
            'item_ids': [10, 20, 30],
            'ratings': [5, 4, 3]
        }
    
    def test_model_initialization(self):
        """Test model initializes with correct dimensions."""
        model = NCFModel(num_users=100, num_items=50, embedding_dim=32)
        assert model.user_embedding.num_embeddings == 100
        assert model.item_embedding.num_embeddings == 50
    
    def test_forward_pass(self, sample_data):
        """Test forward pass produces valid predictions."""
        model = NCFModel(num_users=100, num_items=50)
        predictions = model(sample_data['user_ids'], sample_data['item_ids'])
        assert predictions.shape == (3,)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 5)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Coverage

- Aim for **>80% code coverage**
- All new features must include tests
- Bug fixes should include regression tests

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include type hints
- Provide examples where helpful

### README and Guides

- Update README.md if adding major features
- Add/update relevant documentation in `docs/`
- Include diagrams for complex features (use Mermaid)

### API Documentation

- Update `docs/API_REFERENCE.md` for API changes
- Include request/response examples
- Document error codes and responses

## Pull Request Process

### Before Submitting

**Checklist**:
- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #123

## Changes Made
- Added feature X
- Fixed bug Y
- Updated documentation Z

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: At least one maintainer reviews the code
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves the PR
5. **Merge**: Maintainer merges the PR

### After Merge

- Delete your feature branch
- Update your local repository
- Close related issues

## Issue Reporting

### Bug Reports

**Template**:
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.11]
- Package versions: [run `pip freeze`]

**Additional context**
Any other relevant information.
```

### Feature Requests

**Template**:
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other relevant information.
```

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor graph

## Questions?

- **Documentation**: Check the [docs/](docs/) directory
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/machinelearningAWS-project/discussions)
- **Email**: maintainers@example.com

---

Thank you for contributing to the ML-Powered Product Recommendation System! 🎉
