# Changelog

All notable changes to the MINT TextGraph Library project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-27

### 🎉 Added - Initial Release

#### Core TextGraph Features
- **TextGraph class** with full graph construction capabilities
- **Node types**: Word nodes, Sentence nodes, Claim nodes
- **Basic statistics**: Node/edge counts, word frequency analysis
- **Shared words detection** between context and claim
- **Visualization support** with matplotlib and NetworkX

#### Dependency Parsing Integration
- **Dependency parsing** integration with py_vncorenlp
- **18 dependency relationship types** supported (nmod, vmod, punct, etc.)
- **Word dependency analysis** - find heads and dependents
- **Enhanced statistics** with dependency breakdown
- **Dual edge types**: Structural edges và Dependency edges

#### Advanced Visualization
- **Enhanced visualization** with color-coded node types
- **Dependency-aware visualization** - red dashed lines for dependencies
- **Dependency-only graph view** with labels
- **Customizable figure sizes** and layouts

#### Export & Import
- **GEXF format support** with proper None value handling
- **JSON export** with full metadata
- **Save/Load functionality** with absolute path handling
- **Graph persistence** across sessions

#### Performance & Usability
- **Singleton pattern** for py_vncorenlp model to avoid JVM conflicts
- **Comprehensive error handling** for file operations
- **Detailed documentation** with API reference
- **Demo scripts** for learning and testing

### 🛠️ Technical Details

#### Dependencies
- `py_vncorenlp` for Vietnamese NLP processing
- `networkx` for graph operations
- `matplotlib` for visualization
- `numpy` for numerical operations

#### File Structure
```
FactChecking/
├── README.md                 # Comprehensive documentation
├── CHANGELOG.md             # This file
├── main.py                  # Main demo script
├── demo_dependency.py       # Dependency parsing demo
├── requirements.txt         # Python dependencies
├── text_graph.gexf         # Sample exported graph
└── mint/                   # MINT Library
    ├── __init__.py
    ├── text_graph.py       # Core TextGraph implementation
    └── README.md           # Library-specific docs
```

#### Performance Benchmarks
- **Graph construction**: ~2s for 132 nodes, 382 edges
- **Dependency parsing**: 178 dependency relationships identified
- **Memory usage**: ~200MB for medium-sized texts
- **Export time**: ~1s for GEXF format

### 📊 Statistics Example
With SAWACO fact-checking example:
- **132 total nodes**: 124 words + 7 sentences + 1 claim
- **382 total edges**: 204 structural + 178 dependency
- **30 shared words** between context and claim
- **18 unique dependency types** identified

### 🎯 Fact-checking Applications
- **Semantic similarity** analysis
- **Evidence detection** and linking
- **Contradiction analysis** through dependency structure
- **Feature extraction** for ML models

## [Roadmap] - Future Versions

### [1.1.0] - Planned
- [ ] Named Entity Linking
- [ ] Coreference Resolution
- [ ] Similarity Scoring Algorithms
- [ ] Performance optimizations

### [1.2.0] - Planned
- [ ] Multi-document support
- [ ] Real-time API
- [ ] Machine Learning integration
- [ ] Advanced visualization features

---

**Author**: Hòa Đinh  
**Project**: FactChecking with MINT TextGraph Library 