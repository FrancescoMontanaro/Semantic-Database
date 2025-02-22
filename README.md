# Semantic Database

## Overview
The **Semantic Database** is a document indexing and retrieval system that leverages dense passage retrieval (DPR) and FAISS for efficient semantic search. It allows users to query a collection of documents using natural language, returning the most relevant results based on their semantic similarity.

## Features
- **Semantic Search:** Uses DPR encoders to generate high-quality embeddings for documents and queries.
- **Efficient Indexing:** Utilizes FAISS to store and retrieve vectorized representations efficiently.
- **Automatic Embedding Updates:** Detects changes in documents and updates their embeddings accordingly.
- **Visualization:** Projects document embeddings in 2D using t-SNE for insight into document distribution.
- **Secure Hashing:** Uses cryptographic hashing to track document modifications.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install faiss-cpu transformers numpy tqdm scikit-learn matplotlib
```

### Clone Repository
```bash
git clone https://github.com/your-repo/semantic-database.git
cd semantic-database
```

## Usage
### Initializing the Database
```python
from semantic_database import SemanticDatabase

# Initialize database with the directory containing documents
db = SemanticDatabase(base_directory="path/to/documents")
```

### Creating or Updating the Database
```python
db.create_or_update(destination_path="path/to/save/database")
```

### Loading an Existing Database
```python
db.load("path/to/saved/database")
```

### Querying the Database
```python
results = db.query("What is the capital of France?", num_results=3)
for result in results:
    print(result)
```

### Visualizing the Document Space
```python
db.plot_space()
```

## File Structure
```
semantic_database/
│── cryptography.py  # Handles document hashing
│── utils.py         # Utility functions
│── semantic_database.py  # Main class implementation
│── database.index   # FAISS index file (generated after indexing)
│── documents.json   # Mapping of indexed documents (generated after indexing)
```

## Configuration
The system uses the following pre-trained transformer models:
- **Query Encoder:** `facebook/dpr-question_encoder-single-nq-base`
- **Document Encoder:** `facebook/dpr-ctx_encoder-single-nq-base`

## Error Handling
- If a document is moved or deleted, a warning is logged.
- If the FAISS index or mapping file is corrupted, an error is raised.
- If models are not loaded, the system automatically initializes them.