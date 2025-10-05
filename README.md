# Agentic RAG System

## Overview

**Agentic RAG** (Retrieval-Augmented Generation) is an advanced document question-answering system **designed and implemented from scratch**. It goes beyond traditional RAG by incorporating intelligent agents that reason, plan, and adapt retrieval strategies. The system supports both **Traditional RAG** and **Agentic RAG** modes for fair, scientific comparison.

- **Traditional RAG:** One-shot semantic retrieval + LLM answer synthesis.
- **Agentic RAG:** Multi-agent, iterative retrieval, adaptive strategies, quality assessment, and LLM synthesis.

This project is implemented in Python and supports interactive usage via a **Streamlit web app** and a **Jupyter notebook**.

---

## Features

- **Multi-Agent Orchestration:** Query analysis, retrieval, quality assessment, citation generation, and LLM synthesis.
- **Hybrid Retrieval:** Semantic (FAISS), lexical (BM25), and hybrid search strategies.
- **Iterative Refinement:** Agentic RAG adapts retrieval strategy based on quality metrics.
- **LLM Integration:** Uses HuggingFace models (e.g., Flan-T5, OPT) for answer generation.
- **Fair Comparison:** Both RAG modes use the same LLM, prompt engineering, and evaluation metrics.
- **Comprehensive Evaluation:** Quality metrics include relevance, term overlap, diversity, coverage, and length consistency.
- **Interactive UI:** Streamlit app for querying, comparing, and visualizing results.
- **Extensible:** Modular design for easy extension and experimentation.
- **Built From Scratch:** All orchestration, agent logic, and evaluation are custom-built, not relying on frameworks like CrewAI or LangChain agents.

---

## Architecture

```
User Query
   │
   ├─► QueryAnalysisAgent ──► RetrievalAgent ──► QualityAssessmentAgent ──► CitationAgent ──► LLMGenerator
   │
   └─► (Traditional RAG: Direct semantic retrieval + LLM)
```

- **Traditional RAG:** Semantic search → LLM synthesis.
- **Agentic RAG:** Query analysis → iterative retrieval (semantic, lexical, hybrid) → quality assessment → best results → LLM synthesis.

---

## Setup

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/agentic-rag-system.git
cd agentic-rag-system
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `langchain`, `sentence-transformers`, `faiss-cpu`, `rank-bm25`
- `transformers`, `torch`
- `streamlit`, `plotly`, `pandas`, `numpy`
- `PyPDF2`

### 3. **Download the PDF Document**

Place your target PDF (e.g., *Attention Is All You Need*) in the `data/` folder.

---

## Usage

### **A. Streamlit Web App**

```bash
streamlit run streamlit_app.py
```

- Enter your query about transformers, attention mechanisms, or neural networks.
- Choose **Agentic RAG**, **Traditional RAG**, or **Compare Both**.
- View generated answers, quality scores, citations, and analytics.

### **B. Jupyter Notebook**

Open `agentic_rag.ipynb` for step-by-step code, testing, and performance analysis.

---

## Evaluation Metrics

The system uses a robust multi-factor evaluation for retrieval and answer quality:

| Metric             | Description                                   |
|--------------------|-----------------------------------------------|
| Relevance          | Match between query and retrieved chunks      |
| Term Overlap       | Shared terms between query & results          |
| Diversity          | Variety in pages/content                      |
| Coverage           | How well results cover query intent           |
| Length Consistency | Are chunks of reasonable length?              |
| Response Completeness | Is the answer substantial and well-formed? |

**Overall Score:** Weighted sum of all metrics (0–1 scale).

---

## How It Works

1. **Query Analysis:** Determines query type and recommends retrieval strategy.
2. **Retrieval:** Executes semantic, lexical, or hybrid search.
3. **Quality Assessment:** Scores results for relevance, overlap, diversity, coverage, and length.
4. **Iterative Refinement (Agentic RAG):** Adapts strategy if quality is low.
5. **Citation Generation:** Tracks sources and pages.
6. **LLM Synthesis:** Generates a coherent, bullet-point answer using HuggingFace models.
7. **Comparison:** Both RAG modes use the same LLM and metrics for fair comparison.

---

## Example

**Query:**  
> Compare scaled dot-product attention with additive attention

**Agentic RAG Output:**
```
• Scaled dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

• While for small values of dk the two mechanisms perform similarly, additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

• Dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

• While for small values of dk the two mechanisms perform similarly, additive attention computes the compatibility function using a feed-forward network with a single hidden layer.
```
**Quality Score:** 0.865

**Traditional RAG Output:**
```
• Scaled dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

• While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

• While for small values of dk the two mechanisms perform similarly, addit source 3: query with the corresponding key.

• Dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
```
**Quality Score:** 0.702

---

## Comparison & Results

- **Agentic RAG** uses multi-agent reasoning, iterative retrieval, and quality assessment, resulting in more complete and accurate answers.
- **Traditional RAG** uses single-shot retrieval and may miss context or produce repetitive/incomplete answers.
- **Quality scores and analytics** are displayed for every query.

---

## Extending the System

- **Add new retrieval strategies** by extending `HybridRetriever`.
- **Integrate new LLMs** by updating `HuggingFaceLLMGenerator`.
- **Customize evaluation metrics** in `QualityAssessmentAgent`.
- **Experiment with different documents** by changing the PDF source.

---

## Troubleshooting

- Ensure all dependencies are installed (`pip install -r requirements.txt`).
- Place your PDF in the correct `data/` folder.
- For GPU acceleration, install CUDA-compatible `torch` and `transformers`.

---

## License

MIT License

---

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- HuggingFace Transformers
- LangChain
- Sentence Transformers
- FAISS, BM25

---

## Contact

For questions or contributions, open an issue.
