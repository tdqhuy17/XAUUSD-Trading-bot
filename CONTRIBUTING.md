# Contributing to Trading Bot

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate of others and follow standard open-source community guidelines.

## 🤔 How Can I Contribute?

### Reporting Bugs

Before submitting a bug report, please:
1. Check if the issue has already been reported
2. Use the latest version of the code
3. Collect debug logs if possible

When submitting a bug report, include:
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal code example
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Logs**: Relevant log output

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- **Description**: Clear description of the enhancement
- **Motivation**: Why this would be useful
- **Implementation**: If you have ideas on how to implement it

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🛠️ Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (recommended)

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=trading_bot --cov-report=html

# Run specific test file
pytest tests/test_sac_agent.py -v
```

### Code Style

This project follows:
- **PEP 8**: Python style guide
- **Type Hints**: Use type annotations where possible
- **Docstrings**: Google style docstrings

Format your code before submitting:

```bash
# Format with black
black trading_bot/ scripts/ tests/

# Sort imports
isort trading_bot/ scripts/ tests/

# Check with flake8
flake8 trading_bot/ scripts/ tests/

# Type check with mypy
mypy trading_bot/
```

## 📝 Coding Guidelines

### Code Organization

```
trading_bot/
├── module.py          # Keep modules focused on single responsibility
├── __init__.py        # Export public API
```

### Naming Conventions

- **Classes**: PascalCase (`SACAgent`, `TradingEnv`)
- **Functions/Methods**: snake_case (`select_action`, `get_observation`)
- **Constants**: UPPER_SNAKE_CASE (`SAC_CONFIG`, `MTF_CONFIG`)
- **Private methods**: _leading_underscore (`_soft_update`)

### Documentation

All public functions and classes should have docstrings:

```python
def select_action(self, observation: np.ndarray) -> np.ndarray:
    """
    Select an action given an observation.
    
    Args:
        observation: Environment observation
        
    Returns:
        Action array in range [-1, 1]
    """
    pass
```

### Type Hints

Use type hints for better code clarity:

```python
from typing import Dict, List, Tuple, Optional

def process_data(
    data: pd.DataFrame,
    config: Dict[str, Any],
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Process data with optional normalization."""
    pass
```

### Error Handling

- Use specific exceptions
- Provide helpful error messages
- Handle edge cases gracefully

```python
# Good
if len(data) < min_bars:
    raise ValueError(
        f"Insufficient data: {len(data)} bars, "
        f"need at least {min_bars}"
    )

# Avoid
if len(data) < min_bars:
    raise Exception("Error")
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── test_sac_agent.py      # Unit tests for SAC agent
├── test_environment.py    # Unit tests for environment
├── test_features.py       # Unit tests for features
├── test_data_loader.py    # Unit tests for data loading
└── integration/           # Integration tests
    └── test_training.py
```

### Writing Tests

```python
import pytest
import numpy as np
from trading_bot.sac_agent import SACAgentMTF

class TestSACAgent:
    """Tests for SAC agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        config = {"hidden_dim": 64, "batch_size": 8}
        return SACAgentMTF(config)
    
    def test_select_action_shape(self, agent):
        """Test that action has correct shape."""
        obs = np.zeros((10, 30))
        action = agent.select_action(obs)
        assert action.shape == (1,)
    
    def test_action_in_range(self, agent):
        """Test that actions are in valid range."""
        obs = np.random.randn(10, 30)
        action = agent.select_action(obs)
        assert -1 <= action[0] <= 1
```

## 📦 Project Structure

When adding new features:

1. **Core functionality** → `trading_bot/`
2. **Executable scripts** → `scripts/`
3. **Example code** → `examples/`
4. **Tests** → `tests/`

## 🔒 Security

- Never commit API keys, passwords, or secrets
- Use environment variables for sensitive configuration
- Report security vulnerabilities privately

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Documentation improvements

Thank you for contributing! 🎉