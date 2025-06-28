#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for MINT TextGraph Library
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "MINT TextGraph Library - A library for building and analyzing text graphs from Vietnamese text"

# Read requirements
def read_requirements():
    """Read requirements.txt for dependencies"""
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return [
            'py_vncorenlp',
            'networkx',
            'matplotlib',
            'numpy',
            'openai',
            'python-dotenv',
            'transformers',
            'torch',
            'faiss-cpu',
            'scikit-learn'
        ]

# Get version from mint/__init__.py
def get_version():
    """Get version from mint/__init__.py"""
    here = os.path.abspath(os.path.dirname(__file__))
    version_path = os.path.join(here, 'mint', '__init__.py')
    version = "1.0.0"  # default
    try:
        with open(version_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    break
    except FileNotFoundError:
        pass
    return version

setup(
    name="mint-textgraph",
    version=get_version(),
    author="Hòa Đinh",
    author_email="hoadm.net@gmail.com",
    description="A comprehensive library for building and analyzing text graphs from Vietnamese text with Entity Extraction and Semantic Similarity",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/hoadm-net/FactChecking",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Vietnamese",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "gpu": [
            "faiss-gpu",  # For GPU acceleration
            "torch-audio",  # Additional torch components
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0", 
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "faiss-gpu",
            "torch-audio",
        ]
    },
    entry_points={
        "console_scripts": [
            "mint-demo=mint.cli:main",  # Command line interface (if needed)
        ],
    },
    keywords=[
        "nlp",
        "vietnamese",
        "text-graph", 
        "dependency-parsing",
        "fact-checking",
        "entity-extraction",
        "semantic-similarity",
        "phobert",
        "vncore",
        "graph-analysis"
    ],
    project_urls={
        "Bug Reports": "https://github.com/hoadm-net/FactChecking/issues",
        "Source": "https://github.com/hoadm-net/FactChecking",
        "Documentation": "https://github.com/hoadm-net/FactChecking/blob/main/README.md",
    },
    include_package_data=True,
    package_data={
        "mint": [
            "*.md",
            "*.txt",
        ],
    },
    zip_safe=False,  # Due to model files
) 