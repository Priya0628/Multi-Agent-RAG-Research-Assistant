# 🤖 Multi-Agent RAG Research System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-1.3.0-green.svg)](https://crewai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready RAG pipeline with multi-agent orchestration for automated research synthesis**

A sophisticated knowledge retrieval and synthesis system that combines **Retrieval-Augmented Generation (RAG)** with **multi-agent AI orchestration** using CrewAI. The system ingests documents, creates semantic embeddings, and orchestrates specialized AI agents to produce verified, publication-ready research briefs.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **🔍 Semantic Search** | Dense vector retrieval using SentenceTransformers for accurate context matching |
| **🤖 Multi-Agent Pipeline** | 4 specialized AI agents with distinct roles (Researcher → Fact-Checker → Editor → Publisher) |
| **📊 Local Embeddings** | Free, offline embeddings using `all-MiniLM-L6-v2` (no API costs for retrieval) |
| **✅ Citation Tracking** | Full source attribution and fact verification against original documents |
| **📄 Multi-Format Output** | Generates Markdown briefs and LinkedIn-ready social posts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT RAG ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐       │
│  │  Documents   │  ───►  │  Embeddings  │  ───►  │  ChromaDB    │       │
│  │  (data/*.txt)│        │  (MiniLM-L6) │        │  (Vector DB) │       │
│  └──────────────┘        └──────────────┘        └──────┬───────┘       │
│        INGEST                                           │               │
│  ═══════════════════════════════════════════════════════════════════   │
│        RETRIEVE                                         │               │
│                                                         ▼               │
│  ┌──────────────┐        ┌──────────────────────────────────────────┐  │
│  │  User Query  │  ───►  │  Semantic Search (ANN)                   │  │
│  └──────────────┘        └────────────────────┬─────────────────────┘  │
│                                               │                         │
│  ═══════════════════════════════════════════════════════════════════   │
│        ORCHESTRATE                            │ context                 │
│                                               ▼                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    CrewAI Sequential Pipeline                     │  │
│  │                                                                    │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐       │  │
│  │  │Researcher│ ► │  Fact-  │ ► │  Editor  │ ► │Publisher │       │  │
│  │  │          │   │ Checker │   │          │   │          │       │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘       │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                               │                         │
│                                               ▼                         │
│                                    ┌──────────────────┐                 │
│                                    │  Output Files    │                 │
│                                    │  - brief.md      │                 │
│                                    │  - linkedin.md   │                 │
│                                    └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-rag.git
cd multi-agent-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DIR=vectorstore
```

### Usage

```bash
# Step 1: Add your documents to data/ folder
mkdir -p data
# Place your .txt files in data/

# Step 2: Run document ingestion
python main.py --ingest

# Step 3: Run research query
python main.py "What are the risks of model collapse in AI?"

# Or use interactive mode
python main.py
```

---

## 📁 Project Structure

```
multi-agent-rag/
├── main.py                    # Entry point with CLI interface
├── configs/
│   └── settings.py            # Pydantic configuration management
├── rag/
│   ├── ingest.py              # Document → Embeddings ETL pipeline
│   └── retriever.py           # Semantic search & retrieval
├── crew/
│   ├── agents_rag.py          # CrewAI agent definitions
│   └── orchestrator.py        # Workflow coordination
├── data/                      # Input documents (your .txt files)
├── vectorstore/               # ChromaDB persistent storage
├── artifacts/                 # Generated outputs
│   ├── brief.md               # Research brief (Markdown)
│   └── linkedin_post.md       # Social media content
├── .env                       # Environment variables
└── pyproject.toml             # Project configuration
```

---

## 🤖 Agent Roles

| Agent | Role | Responsibility |
|-------|------|----------------|
| **Researcher** | Research Analyst | Synthesize findings from retrieved context |
| **Fact-Checker** | Verification Specialist | Verify claims against sources, add citations |
| **Editor** | Content Editor | Refine prose for clarity and structure |
| **Publisher** | Publishing Specialist | Format for Markdown and social media |

---

## 🧠 Technical Deep Dive

### RAG Pipeline

1. **Ingestion** (`rag/ingest.py`)
   - Load documents from `data/` folder
   - Apply sliding window chunking (500 chars, 100 overlap)
   - Generate dense embeddings using SentenceTransformers
   - Store in ChromaDB with source metadata

2. **Retrieval** (`rag/retriever.py`)
   - Embed user query with same model
   - Approximate Nearest Neighbor search (HNSW algorithm)
   - Return top-k chunks with relevance scores

### Multi-Agent Orchestration

- **Pattern**: Sequential Processing (`Process.sequential`)
- **Framework**: CrewAI v1.3.0
- **LLM**: GPT-4o-mini (temperature=0 for determinism)

Each task receives output from the previous agent, enabling progressive refinement while maintaining source attribution.

---

## 📊 Example Output

### Research Brief (`artifacts/brief.md`)

```markdown
# Model Collapse in Large Language Models

**Date:** 2025-01-15

## Executive Summary

Model collapse is a phenomenon where AI models trained on synthetic data 
progressively degrade in quality... (Source: model_collapse_paper.txt)

## Key Takeaways

- Model collapse occurs when AI trains on its own outputs
- Quality degradation is cumulative across generations
- Maintaining original training data is critical

## Sources
- model_collapse_paper.txt
- ai_risks_overview.txt
```

### LinkedIn Post (`artifacts/linkedin_post.md`)

```
🔬 New research on Model Collapse in AI

Training AI on synthetic data leads to quality degradation. 
Here's why data provenance matters...

#AI #MachineLearning #Research #LLM
```

---

## 🔧 Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | LLM model for agents | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Local embedding model | `all-MiniLM-L6-v2` |
| `CHROMA_DIR` | Vector DB storage path | `vectorstore` |

---

## 🧪 Testing

```bash
# Test ingestion
python -c "from rag.ingest import build_vectorstore; build_vectorstore('data')"

# Test retrieval
python -c "from rag.retriever import retrieve; print(retrieve('test query', k=3))"

# Test agents
python -c "from crew.agents_rag import AGENTS; print(list(AGENTS.keys()))"

# Full system test
python main.py "What is artificial intelligence?"
```

---

## 📈 Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Document Ingestion | O(n × d) | n=documents, d=avg length |
| Embedding Generation | O(n × c) | n=chunks, c=chunk length |
| Semantic Search | O(log k) | HNSW approximate search |
| Agent Execution | O(4) | 4 sequential agent calls |

---

## 🔮 Future Enhancements

- [ ] Hierarchical agent orchestration
- [ ] PDF native ingestion with OCR
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Custom embedding fine-tuning
- [ ] Evaluation metrics dashboard

---

## 📚 References

- [CrewAI Documentation](https://docs.crewai.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [RAG Paper - Lewis et al.](https://arxiv.org/abs/2005.11401)

ngChain, and OpenAI*
