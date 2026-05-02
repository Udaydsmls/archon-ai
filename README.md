# Archon

A production-grade multi-agent system that autonomously researches a topic, retrieves relevant documents, synthesizes findings, and self-critiques its output — served via a REST API with real-time streaming.

---

## Architecture

The system is built on a **LangGraph StateGraph** where each node is a specialist agent. Agents communicate exclusively through a shared typed state, with no direct coupling between them.

```
User Query
    │
    ▼
ResearchAgent  ──── ReAct loop (web search + URL scraping)
    │
    ▼
RAGAgent       ──── Retrieves relevant chunks from vector store
    │
    ▼
SynthesisAgent ──── Drafts a structured report (with code execution)
    │
    ▼
CriticAgent    ──── Scores draft 0–10, returns structured feedback
    │
    ├── score < threshold AND cycles < max ──► SynthesisAgent (revise)
    │
    └── score ≥ threshold OR max cycles reached ──► Final Report
```

---

## Agent Patterns

| Agent | Pattern | Model |
|---|---|---|
| `ResearchAgent` | ReAct (Reason + Act loop) | claude-sonnet-4-6 |
| `RAGAgent` | Retrieval-Augmented Generation | claude-sonnet-4-6 |
| `SynthesisAgent` | Tool-augmented generation | claude-sonnet-4-6 |
| `CriticAgent` | Self-reflection with structured JSON output | claude-opus-4-7 |

### ReAct Loop
The `ResearchAgent` iterates: **Think → Tool Call → Observe → Think** until it reaches a final answer. Tools available: `web_search` (Tavily) and `scrape_url` (BeautifulSoup).

### Self-Reflection
The `CriticAgent` scores each draft on factual accuracy, completeness, coherence, and citation quality. If the score is below the configured threshold, the draft is returned to `SynthesisAgent` with specific improvement instructions.

### Hierarchical Delegation
The `Orchestrator` owns the LangGraph and delegates to each agent in sequence. Routing decisions (revise vs. finalize) are made by the orchestrator based on critique score and cycle count — agents have no awareness of each other.

---

## RAG Pipeline

```
Document / URL / File
        │
        ▼
   TextChunker        Word-window chunking with configurable overlap
        │
        ▼
   VectorStore        ChromaDB + SentenceTransformer (all-MiniLM-L6-v2)
        │
        ▼
  HybridRetriever     Semantic similarity search, top-k results
        │
        ▼
    RAGAgent          Extracts and structures relevant context
```

Ingest documents before running queries:

```python
from backend.rag.ingestor import DocumentIngestor
from backend.rag.vector_store import VectorStore
from backend.rag.chunker import TextChunker

ingestor = DocumentIngestor(VectorStore(), TextChunker())
ingestor.ingest_url("https://example.com/article")
ingestor.ingest_file("./docs/paper.txt")
```

---

## LLM-Native Metrics

Every agent invocation is tracked automatically. Metrics are accumulated in state and queryable via the API.

| Metric | Description |
|---|---|
| `tokens/sec` | Output tokens per second per agent call |
| `cost_usd` | Per-call cost based on model pricing (input/output/cache) |
| `cache_read_tokens` | Prompt cache hits (system prompts are cached via `cache_control`) |
| `cache_write_tokens` | Tokens written to prompt cache |
| `tool_calls` | Number of tool invocations per agent run |
| `reflection_cycles` | Number of critique-revise loops before finalization |

Retrieve metrics for a run:

```bash
GET /api/v1/run/{run_id}/metrics
```

---

## Prompt Engineering

All prompts are versioned in [`backend/agents/prompts.py`](backend/agents/prompts.py). System prompts are injected with `cache_control: ephemeral` to reduce cost on repeated calls.

```python
PROMPTS = {
    "research_system_v1": "...",
    "critic_system_v1": "...",
    ...
}
```

Updating a prompt bumps the version key (`_v1` → `_v2`), preserving the previous version for rollback.

---

## Fine-Tuning

The `finetuning/` module generates synthetic training data and evaluates a fine-tuned critic model against the baseline.

```bash
# Generate labeled (draft, score, feedback) pairs
python -m backend.finetuning.data_generator

# Evaluate fine-tuned model vs baseline
python -m backend.finetuning.evaluator
```

The goal is to replace expensive `claude-sonnet` critique calls with a smaller fine-tuned model, reducing cost per run.

---

## API

### Authentication

```bash
POST /api/v1/auth/token
{"api_key": "dev-key-change-in-production"}
# returns {"access_token": "...", "token_type": "bearer"}
```

### Run a Query (Synchronous)

```bash
POST /api/v1/run
Authorization: Bearer <token>
{"query": "What are the latest advances in quantum computing?"}
```

### Run a Query (Streaming)

```bash
POST /api/v1/run/stream
Authorization: Bearer <token>
{"query": "What are the latest advances in quantum computing?"}
```

Returns Server-Sent Events — one event per agent node as the pipeline executes:

```
data: {"node": "research_agent", "update": {"research_results": [...]}}
data: {"node": "rag_agent", "update": {"rag_context": [...]}}
data: {"node": "synthesis_agent", "update": {"draft": "..."}}
data: {"node": "critic_agent", "update": {"critique_score": 8.2, ...}}
data: {"node": "__end__", "update": {}}
```

### Get Run Metrics

```bash
GET /api/v1/run/{run_id}/metrics
Authorization: Bearer <token>
```

---

## Project Structure

```
backend/
├── config.py                   # Pydantic-settings environment config
├── main.py                     # FastAPI app entry point
├── state/
│   └── schema.py               # Shared LangGraph AgentState (TypedDict)
├── metrics/
│   ├── models.py               # TokenUsage, AgentMetric dataclasses
│   └── tracker.py              # Aggregates metrics across agents
├── tools/
│   ├── base.py                 # Tool Protocol (structural subtyping)
│   ├── web_search.py           # Tavily web search
│   ├── url_scraper.py          # BeautifulSoup URL scraper
│   └── code_executor.py        # Sandboxed Python subprocess runner
├── rag/
│   ├── chunker.py              # Word-window text chunker
│   ├── vector_store.py         # ChromaDB adapter
│   ├── retriever.py            # Semantic similarity retrieval
│   └── ingestor.py             # Text / file / URL ingestion pipeline
├── agents/
│   ├── base.py                 # BaseAgent ABC
│   ├── prompts.py              # Versioned prompt registry
│   ├── research_agent.py       # ReAct web research
│   ├── rag_agent.py            # RAG context extraction
│   ├── critic_agent.py         # Structured self-reflection scoring
│   ├── synthesis_agent.py      # Report drafting
│   └── orchestrator.py         # LangGraph StateGraph + routing
├── api/
│   ├── models.py               # Pydantic request/response schemas
│   ├── auth.py                 # JWT issuance and validation
│   ├── streaming.py            # SSE event serializer
│   └── routes.py               # FastAPI route handlers
└── finetuning/
    ├── data_generator.py       # Synthetic training data generation
    └── evaluator.py            # Fine-tuned vs baseline evaluation
```

---

## Setup

### Local

```bash
git clone https://github.com/Udaydsmls/agentic-ai.git
cd agentic-ai
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python -m backend.main
```

### Docker

```bash
cp .env.example .env          # add your API keys
docker-compose up --build
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key | required |
| `TAVILY_API_KEY` | Tavily search API key | required |
| `PRIMARY_MODEL` | Model for research/synthesis agents | `claude-sonnet-4-6` |
| `CRITIC_MODEL` | Model for critic agent | `claude-opus-4-7` |
| `MAX_REFLECTION_CYCLES` | Max critique-revise iterations | `3` |
| `CRITIQUE_PASS_THRESHOLD` | Minimum score to finalize report | `7.0` |
| `CHROMA_PERSIST_DIR` | Path for ChromaDB persistence | `./chroma_db` |
| `DATABASE_URL` | SQLite path for LangGraph checkpointing | `sqlite:///./runs.db` |
| `JWT_SECRET` | Secret key for JWT signing | change in production |

---

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Anthropic Claude (claude-sonnet-4-6, claude-haiku-4-5)
- **Vector Store:** ChromaDB + SentenceTransformers
- **Web Search:** Tavily
- **API:** FastAPI + Uvicorn
- **Auth:** JWT (python-jose)
- **Containerization:** Docker + docker-compose
