# Archon

A production-grade multi-agent system that autonomously researches a topic, retrieves relevant documents, synthesizes findings, and self-critiques its output — served via a REST API with real-time streaming.

---

## Architecture

The system is built on a **LangGraph StateGraph** where each node is a specialist agent. Agents communicate exclusively through a shared typed state, with no direct coupling between them.

```
User Query (text or multimodal)
    │
    ▼
ResearchAgent  ──── ReAct loop (web search + URL scraping + vision)
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
    └── score ≥ threshold OR max cycles reached ──► Safety Validator ──► Final Report
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
The `ResearchAgent` iterates: **Think → Tool Call → Observe → Think** until it reaches a final answer. Tools available: `web_search` (Tavily), `scrape_url` (BeautifulSoup), and `extract_from_file` (vision).

### Self-Reflection
The `CriticAgent` scores each draft on factual accuracy, completeness, coherence, and citation quality. If the score is below the configured threshold, the draft is returned to `SynthesisAgent` with specific improvement instructions.

### Hierarchical Delegation
The `Orchestrator` owns the LangGraph and delegates to each agent in sequence. Routing decisions (revise vs. finalize) are made by the orchestrator based on critique score and cycle count — agents have no awareness of each other.

---

## Upgrades

### A — Pluggable Vector Store

The vector store backend is swappable via the `VECTOR_STORE_PROVIDER` environment variable with no code changes.

| Value | Backend |
|---|---|
| `local` (default) | ChromaDB (local persistence) |
| `pinecone` | Pinecone serverless index |
| `weaviate` | Weaviate v4 with native hybrid search |

All providers implement `VectorStoreProvider` (`upsert`, `similarity_search`, `hybrid_search`). Enable BM25 + dense **Hybrid Retrieval** via Reciprocal Rank Fusion with `USE_HYBRID_RETRIEVAL=true`.

### B — Evaluation Harness

A standalone evaluation suite that black-box tests the full pipeline against a 20-question golden dataset.

```bash
python -m evaluation.run_eval --suite ragas     # faithfulness, relevancy, precision, recall
python -m evaluation.run_eval --suite deepeval  # hallucination, bias, answer relevancy
python -m evaluation.run_eval --suite all       # exits with code 1 if any metric fails
```

Results are saved to `evaluation/results/`. A GitHub Actions workflow is available for manual runs via **Actions → Evaluation Suite → Run workflow**.

### C — LangSmith Tracing

Full prompt/response lineage tracing across all agent hops with zero agent code changes. Enable with:

```env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=archon-ai
```

LangSmith hooks into LangGraph automatically via the LangChain callback system.

### D — Guardrails AI Safety Layer

Post-processing validation on all final reports before they reach the caller. Enable with `GUARDRAILS_ENABLED=true`.

| Guard | Behaviour |
|---|---|
| `DetectPII` | Redacts emails, phones, SSNs, credit cards |
| `ToxicLanguage` | Flags hate speech patterns |
| `RestrictToTopic` | Warns if output is off-topic (optional allowlist) |

Violations are appended as `safety_warnings` in the response. When disabled, the validator is a no-op passthrough.

### E — Batch Ingestion

Ingest hundreds of documents cost-efficiently using the Anthropic Messages Batch API:

```bash
POST /api/v1/ingest/batch
{"urls": ["https://..."], "file_paths": ["/path/to/doc.txt"]}
```

Documents are chunked, submitted as a single batch job (50% cost reduction), and summaries + original chunks are upserted to the vector store on completion.

### F — Multimodal Input

Submit images and PDFs alongside text queries:

```bash
POST /api/v1/run/multimodal   (multipart/form-data)
fields: query (text), files (images/PDFs)
```

The `VisionTool` uses Claude's vision capability to extract structured text from each file, which is prepended to the research context before the ReAct loop begins. Existing text-only routes are unchanged.

### G — DynamoDB Multi-Tenant Storage

Production-grade persistent storage with RBAC, switchable via `STORAGE_BACKEND=dynamo`.

| Table | Partition Key | Sort Key | Contents |
|---|---|---|---|
| `archon_runs` | `tenant_id` | `run_id` | Full run records |
| `archon_tenants` | `tenant_id` | — | RBAC: `allowed_agents`, `rate_limit_per_hour` |

SQLite remains the default for local development. DynamoDB is opt-in for production.

---

## RAG Pipeline

```
Document / URL / File
        │
        ▼
   TextChunker           Word-window chunking with configurable overlap
        │
        ▼
  VectorStoreProvider    ChromaDB (local) | Pinecone | Weaviate
        │
        ▼
  SemanticRetriever      Dense similarity search
  HybridRetriever        BM25 + dense via Reciprocal Rank Fusion (opt-in)
        │
        ▼
    RAGAgent             Extracts and structures relevant context
```

---

## LLM-Native Metrics

Every agent invocation is tracked automatically.

| Metric | Description |
|---|---|
| `tokens/sec` | Output tokens per second per agent call |
| `cost_usd` | Per-call cost based on model pricing (input/output/cache) |
| `cache_read_tokens` | Prompt cache hits (system prompts cached via `cache_control`) |
| `cache_write_tokens` | Tokens written to prompt cache |
| `tool_calls` | Number of tool invocations per agent run |
| `reflection_cycles` | Critique-revise loops before finalization |

---

## API Reference

### Authentication
```bash
POST /api/v1/auth/token
{"api_key": "dev-key-change-in-production"}
```

### Run (Synchronous)
```bash
POST /api/v1/run
{"query": "What are the latest advances in quantum computing?"}
```

### Run (Streaming SSE)
```bash
POST /api/v1/run/stream
{"query": "What are the latest advances in quantum computing?"}
```

### Run (Multimodal)
```bash
POST /api/v1/run/multimodal   (multipart/form-data)
query=... + files=...
```

### Batch Ingest
```bash
POST /api/v1/ingest/batch
{"urls": ["https://..."], "file_paths": []}
```

### Metrics
```bash
GET /api/v1/run/{run_id}/metrics
```

---

## Project Structure

```
backend/
├── config.py
├── main.py
├── state/schema.py
├── metrics/
│   ├── models.py
│   └── tracker.py
├── tools/
│   ├── base.py
│   ├── web_search.py
│   ├── url_scraper.py
│   ├── code_executor.py
│   └── vision_tool.py
├── rag/
│   ├── chunker.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── ingestor.py
│   ├── batch_ingestor.py
│   └── providers/
│       ├── base.py
│       ├── chroma_provider.py
│       ├── pinecone_provider.py
│       ├── weaviate_provider.py
│       └── factory.py
├── agents/
│   ├── base.py
│   ├── prompts.py
│   ├── research_agent.py
│   ├── rag_agent.py
│   ├── critic_agent.py
│   ├── synthesis_agent.py
│   └── orchestrator.py
├── safety/
│   ├── config.py
│   └── guards.py
├── storage/
│   ├── base.py
│   ├── sqlite_backend.py
│   ├── dynamo_backend.py
│   └── factory.py
├── api/
│   ├── models.py
│   ├── auth.py
│   ├── streaming.py
│   └── routes.py
└── finetuning/
    ├── data_generator.py
    └── evaluator.py

evaluation/
├── golden_dataset.json
├── ragas_eval.py
├── deepeval_eval.py
└── run_eval.py

.github/workflows/eval.yml
```

---

## Setup

### Local

```bash
git clone https://github.com/Udaydsmls/archon-ai.git
cd archon-ai
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python -m backend.main
```

### Docker

```bash
cp .env.example .env
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
| `VECTOR_STORE_PROVIDER` | Vector store backend | `local` |
| `USE_HYBRID_RETRIEVAL` | Enable BM25 + dense RRF retrieval | `false` |
| `STORAGE_BACKEND` | Persistence backend | `sqlite` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `GUARDRAILS_ENABLED` | Enable safety validation layer | `false` |

See `.env.example` for the full list.

---

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Anthropic Claude (claude-sonnet-4-6, claude-opus-4-7)
- **Vector Store:** ChromaDB / Pinecone / Weaviate (pluggable)
- **Retrieval:** SentenceTransformers + BM25 (rank-bm25) + RRF
- **Evaluation:** RAGAS + DeepEval
- **Tracing:** LangSmith
- **Safety:** Guardrails AI
- **Storage:** SQLite / DynamoDB (pluggable)
- **API:** FastAPI + Uvicorn
- **Auth:** JWT (python-jose)
- **Batch Processing:** Anthropic Messages Batch API
- **Containerization:** Docker + docker-compose
