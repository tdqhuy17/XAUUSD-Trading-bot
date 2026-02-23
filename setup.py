"""
Setup script for Trading Bot package
Multi-Timeframe SAC Reinforcement Learning Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                # Skip lines with URLs (special pip syntax)
                if "--index-url" not in line and "--extra-index-url" not in line:
                    requirements.append(line)

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

setup(
    name="trading-bot",
    version="3.0.0",
    author="Trading Bot Contributors",
    author_email="",
    description="Multi-Timeframe SAC Reinforcement Learning Trading System for XAUUSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/trading-bot",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "scripts"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "mt5": ["MetaTrader5>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "trading-train=scripts.train:main",
            "trading-backtest=scripts.backtest:main",
            "trading-live=scripts.run_live:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Machine Learning",
        "Typing :: Typed",
    ],
    keywords=[
        "trading",
        "reinforcement-learning",
        "sac",
        "soft-actor-critic",
        "machine-learning",
        "deep-learning",
        "pytorch",
        "finance",
        "gold",
        "xauusd",
        "algorithmic-trading",
        "multi-timeframe",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/trading-bot/issues",
        "Source": "https://github.com/yourusername/trading-bot",
        "Documentation": "https://github.com/yourusername/trading-bot#readme",
    },
)