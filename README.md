# Multi-Step Reasoning Agent with Verification

A structured reasoning system built with **Pydantic AI** that breaks complex queries into verifiable reasoning steps. Supports **OpenAI**, **Anthropic (Claude)**, and **LM Studio (local models)** as interchangeable backends. Optionally grounds answers in a local document corpus via a **RAG** pipeline backed by ChromaDB.

## Pipeline

```
User Query
    ↓
[if RAG_ENABLED] VectorStore.retrieve() ← ChromaDB (.chromadb/)
    ↓
Reasoning Agent → tools: retrieve, search, calculator
    ↓
ReasoningChain (structured steps + final answer + confidence)
    ↓
Verification Agent (independent, no tools)
    ↓
VerificationReport (per-step audit + overall score)
```

## Project Structure

```
pydantic_sample/
├── agents/
│   ├── reasoning.py        # Multi-step reasoning agent + exponential backoff retry
│   └── verifier.py         # Independent verification agent + exponential backoff retry
├── config/
│   └── settings.py         # Provider config + RAG settings
├── data/
│   └── sample_docs/        # Demo corpus (.txt files) for RAG
├── schemas/
│   ├── reasoning.py        # ToolCall, ReasoningStep, ReasoningChain
│   ├── retrieval.py        # RetrievedChunk, RetrievalResult
│   └── verification.py     # StepVerification, VerificationReport
├── scripts/
│   └── ingest.py           # CLI: chunk + embed + index docs → ChromaDB
├── tools/
│   ├── calculator.py       # Safe math via AST parsing (no eval)
│   ├── retriever.py        # VectorStore (ChromaDB), chunk_text, is_empty, retrieve
│   └── search.py           # Web search with demo fallback + warning log
├── tests/                  # Full test suite (no real API calls, 92% coverage)
├── main.py                 # CLI entry point + structured error handling
└── .env.example
```

## Setup

```bash
conda create -n pydantic_sample python=3.12 -y
conda activate pydantic_sample
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

```env
# Pick one: openai | anthropic | lmstudio
LLM_PROVIDER=openai

OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o

ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-20250514

LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=your-local-model-name

# RAG (off by default)
RAG_ENABLED=false
VECTOR_STORE_PATH=.chromadb
EMBEDDING_MODEL=text-embedding-3-small
```

## Usage

```bash
python main.py "What is 15% of France's GDP?"
python main.py "query" --provider anthropic
python main.py "query" --no-verify
python main.py "query" --verbose

# RAG (index docs first)
python scripts/ingest.py
RAG_ENABLED=true python main.py "query"
```

**CLI flags:** `--provider [openai|anthropic|lmstudio]`, `--no-verify`, `--verbose`

## Architecture

### Schemas

| Model | Fields |
|-------|--------|
| `ToolCall` | `tool_name`, `tool_input`, `tool_output` |
| `ReasoningStep` | `step_number`, `description`, `reasoning`, `tool_calls`, `conclusion` |
| `ReasoningChain` | `query`, `steps` (min 1), `final_answer`, `confidence` (0.0–1.0) |
| `StepVerification` | `step_number`, `is_valid`, `issues`, `severity` (none/low/medium/high/critical) |
| `VerificationReport` | `chain_is_valid`, `overall_score`, `step_verifications`, `logical_errors`, `potential_hallucinations`, `completeness_issues`, `summary` |
| `RetrievedChunk` | `text`, `source`, `chunk_index`, `score` (0.0–1.0) |
| `RetrievalResult` | `query`, `chunks`; `as_context_string()` formats for LLM injection |

### Tools

| Tool | Description |
|------|-------------|
| `calculator` | AST-based safe eval — whitelist operators only, no `eval()` |
| `search` | DuckDuckGo async HTTP, demo fallback when no API key; emits `WARNING` log when falling back |
| `retrieve` | ChromaDB vector search; returns graceful message when RAG disabled; warns if corpus is empty |

### RAG Pipeline

**Indexing** (`scripts/ingest.py`):

1. Read `.txt` files from `data/sample_docs/`
2. `chunk_text()` splits each file into **400-char chunks with 50-char overlap** (character-level, not semantic)
3. Chunk IDs are `SHA-256(source:index:first_40_chars)` — deterministic, so re-indexing is idempotent (upsert)
4. Chunks are embedded and persisted to ChromaDB at `VECTOR_STORE_PATH`

**Embedding — auto-selected at runtime:**

| Condition | Embedder |
|-----------|----------|
| Valid `OPENAI_API_KEY` | OpenAI `text-embedding-3-small` (cloud) |
| Missing or placeholder key (`sk-...`) | ChromaDB `DefaultEmbeddingFunction` (offline, sentence-transformers) |

`resolve_openai_key()` treats any key ending with `...` as absent, ensuring consistent fallback between `ingest.py` and `main.py`.

**Retrieval** (at query time):

1. `main.py` loads `VectorStore` from `VECTOR_STORE_PATH` when `RAG_ENABLED=true`
2. `VectorStore.is_empty()` is checked immediately — a `WARNING` with a remediation hint is logged if the corpus is empty (run `python scripts/ingest.py` to index documents)
3. `VectorStore` passed into `ReasoningDeps`; `None` disables RAG without any branching in agent logic
4. LLM calls `retrieve_tool(query)` → `VectorStore.retrieve(query, top_k=3)`
5. ChromaDB returns **L2 distances**, converted to `[0, 1]` relevance score: `score = 1.0 / (1.0 + L2_distance)`
6. Top-3 chunks formatted via `as_context_string()` and returned as tool output string

### Agent Design

- **Factory pattern** — `create_reasoning_agent(model)` accepts any model instance (including `TestModel`)
- **Dependency injection** — `ReasoningDeps(search_api_key, vector_store)` passed via `RunContext`
- **Independent verification** — verifier agent receives `chain.model_dump_json()` as input; no shared state with reasoning agent
- **Structured output retries** — both agents retry up to 3 times on output validation failure (pydantic-ai `retries=3`)
- **Transient error retries** — `_run_with_backoff()` wraps each `agent.run()` call with up to 3 attempts and exponential delays (1s → 2s → 4s); `ValueError` (config errors) is re-raised immediately without retrying
- **Provider abstraction** — `Settings.get_model()` returns a unified `pydantic_ai.Model`; LM Studio uses the OpenAI-compatible interface
- **Structured error handling** — `main()` catches `ValueError` (config), `httpx.NetworkError` (connectivity), and all other exceptions separately, printing user-friendly messages with remediation hints

### LM Studio Notes

Requires a model with tool calling support (Qwen2.5, Llama 3.1+, Mistral). Not all local models support structured output reliably — increase retries or use a larger model if validation errors occur.

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=. --cov-report=term-missing
```

All tests use pydantic-ai's `TestModel` — no real API calls required.

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Provider switching, API key validation, RAG settings |
| `test_schemas.py` | Pydantic validation rules, serialization round-trips |
| `test_tools.py` | Calculator injection safety, search demo fallback, demo mode warning log, real mode no-warning |
| `test_retriever.py` | `chunk_text` edge cases, `VectorStore` in-memory add/retrieve/score, `is_empty()` |
| `test_reasoning.py` | Agent factory, tool registration, `ReasoningDeps` with `VectorStore` |
| `test_verifier.py` | Verifier factory, no-tools assertion, `TestModel` integration |
| `test_main.py` | CLI args, output formatting, error handling (ValueError/NetworkError/Exception), empty corpus warning |

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic-ai` | Agent framework with structured output |
| `pydantic-settings` | Environment-based configuration |
| `python-dotenv` | `.env` file loading |
| `httpx` | Async HTTP client (search tool) |
| `chromadb` | Local vector store for RAG |
| `pytest` + `pytest-asyncio` + `pytest-cov` | Testing |
