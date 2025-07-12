# 🤖 Obsidian Assistant RAG

An advanced Retrieval-Augmented Generation (RAG) system designed specifically for Obsidian vaults. This system provides intelligent, context-aware responses by leveraging your markdown notes through sophisticated vector search and multi-stage retrieval processing.

## ✨ Features

- **Advanced Retrieval**: Multi-stage processing with query expansion, semantic search, and reranking
- **Obsidian Integration**: Processes wikilinks `[[page]]`, tags `#tag`, headers, and metadata
- **Fast LLM**: Groq API with llama-3.3-70b-versatile model
- **Vector Search**: FAISS database with sentence-transformers embeddings
- **Web Interface**: Interactive Streamlit chat application
- **Smart Scoring**: Combines semantic similarity, keyword matching, and diversity filtering

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [uv package manager](https://github.com/astral-sh/uv) (recommended)
- Groq API key
- Obsidian vault with markdown files

### Installation

1. **Clone and install**
   ```bash
   git clone <repository-url>
   cd obsidian-assintant-rag
   uv sync
   ```

2. **Configure environment**
   ```bash
   cp environment.env .env
   # Edit .env with your settings
   ```

3. **Set required variables in .env**
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   OBSIDIAN_FOLDER=/path/to/your/obsidian/vault
   ```

4. **Run the application**
   ```bash
   uv run streamlit run app/app.py
   ```

5. **Access at http://localhost:8501**

## ⚙️ Configuration

### Getting API Key
1. Visit [Groq Console](https://console.groq.com/keys)
2. Create account and generate API key
3. Add to your `.env` file

### Vault Setup
- Point `OBSIDIAN_FOLDER` to your Obsidian vault directory
- Ensure it contains `.md` files
- System processes all markdown files recursively

## 📖 Usage

### Web Interface
1. Start app with `streamlit run app/app.py`
2. Check vault/database status in sidebar
3. Toggle RAG mode for document-assisted responses
4. Chat with your documents using natural language

### CLI Interface
```bash
uv run python app/src/groq_chat.py
```

Commands: `/help`, `/quit`, `/clear`, `/rebuild`, `/info`, `/rag on/off`

### Query Examples
- "What are my notes about machine learning?"
- "Summarize my project planning documents"
- "Find information about #python tags"
- "What does [[Important Note]] contain?"

## 🏗️ Architecture

The system follows this pipeline:
1. **Document Ingestion**: Markdown files processed and chunked
2. **Vectorization**: Text converted to embeddings
3. **Index Storage**: Vectors stored in FAISS with metadata
4. **Query Processing**: User queries expanded and enhanced
5. **Retrieval**: Multi-stage search with semantic and keyword matching
6. **Reranking**: Results scored using multiple criteria
7. **Response Generation**: Groq LLM generates contextual responses

## 🔧 Advanced Configuration

### Retrieval Settings
Customize in `groq_chat.py`:
```python
retrieval_config = RetrievalConfig(
    initial_k=15,              # Initial candidates
    final_k=5,                 # Final results
    enable_reranking=True,     # Enable reranking
    diversity_threshold=0.7,   # Diversity filtering
    boost_related_chunks=True, # Boost related content
)
```

### Scoring Weights
- Semantic Similarity: 40%
- Keyword Matching: 25%
- Diversity Score: 15%
- Related Chunk Boost: 10%
- Header Matching: 10%

## 🗂️ Project Structure

```
obsidian-assintant-rag/
├── app/
│   ├── app.py                 # Streamlit interface
│   └── src/
│       ├── __init__.py        # Package initialization
│       ├── groq_chat.py       # Main chat engine
│       ├── retrieval.py       # Retrieval system
│       ├── chunking.py        # Document chunking
│       ├── vector_db.py       # Vector database operations
│       ├── vectorizer.py      # Document vectorization
│       └── document_processor.py # Text processing & metadata
├── assets/
│   └── vector_db/             # FAISS storage
│       ├── faiss_index.bin    # Vector index
│       └── metadata.pkl       # Document metadata
├── environment.env            # Config template
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Lock file
└── README.md
```

## 🛠️ Troubleshooting

**Vector Database Not Found**
- Ensure `OBSIDIAN_FOLDER` points to directory with `.md` files
- Use "Rebuild Vector DB" button

**API Errors**
- Check `GROQ_API_KEY` in `.env`
- Verify key validity and credits

**Empty Results**
- Try different keywords
- Rebuild database
- Check vault contains relevant content

**Memory Issues**
- Process large vaults in smaller batches
- Monitor RAM usage during vectorization

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for web interface

---

**Happy note querying! 🚀**