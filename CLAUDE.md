# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research assistant that accelerates progression from domain curiosity to testable hypothesis. Four-stage pipeline: **Orient** (domain mapping) → **Retrieve** (select kb content) → **Distill** (reasoning extraction) → **Translate** (corpus-aware hypothesis generation).

Content ingestion is handled by the `knowledge-base` repo (`kb ingest`). This repo consumes `kb.db` read-only.

## Development Setup

Uses **uv** for package management (not pip). Python 3.13.

```bash
uv sync          # install dependencies
uv run ra --help # CLI entry point
```

### Required environment variables (`.env`, gitignored)
- `ANTHROPIC_API_KEY` — for LLM calls (Distill, Translate)
- `OPENAI_API_KEY` — for insight embedding to chroma (optional, enables corpus-aware Translate)
- `KB_DB_PATH` — path to kb.db (default: `~/.knowledge-base/kb.db`)
- `CHROMA_PERSIST_DIR` — path to ChromaDB (default: `~/.knowledge-base/chroma`)
- `INSIGHT_EMBEDDING_MODEL` — OpenAI model (default: `text-embedding-3-small`)

Config in `src/research_assistant/config.py` — defaults DB to `~/.research-assistant/ra.db`.

## Testing

```bash
uv run pytest                        # run all tests
uv run pytest tests/test_foo.py      # single file
uv run pytest -k "test_name"         # single test by name
```

All LLM, OpenAI, and ChromaDB calls are mocked in tests. No API keys needed.

## CLI Commands

### Pipeline stages
```bash
ra orient --domain "fed_rate_decisions" --market kalshi --known-domains "sports,politics"
ra retrieve --domain nfl [--trust-tier core,supplementary] [--analyst barrett,jj] [--since 2025-01-01] [--dry-run]
ra distill --domain nfl [--batch-id <uuid>] [--mode both] [--limit 10]
ra distill --domain nfl --from-kb                    # direct KB path (bypasses batch)
ra translate --domain nfl --mode explore              # corpus-aware (default)
ra translate --seed-insight-id <id> --domain nfl      # anchor on specific insight
ra translate --domain nfl --domain-registry <path>    # with metrics catalog
ra reembed --domain nfl                               # retry failed insight embeddings
```

### Inspection & export
```bash
ra status --domain fed_rate_decisions
ra list insights --domain fed_rate_decisions [--type framework|claim|observation] [--status active|merged|discarded]
ra list hypotheses --domain fed_rate_decisions [--status draft|accepted|rejected|tested]
ra show domain --domain fed_rate_decisions
ra show insight --id <insight-id>
ra show hypothesis --id <hypothesis-id>
ra export --hypothesis-id <id> --format json
ra export --hypothesis-id <id> --domain-registry <path> --output <path>
```

### Migration (from pre-refactor ra.db)
```bash
ra migrate to-kb-ownership --dry-run   # preview
ra migrate to-kb-ownership             # execute
```

## Architecture

- **Package layout**: `src/research_assistant/` (src layout, hatchling build)
- **Entry point**: `ra` → `research_assistant.cli:cli` (Click)
- **DB**: SQLite via `db.py` — `ra.db` stores domain_brief, insight, hypothesis, retrieval_batch, insight_embedding. Schema auto-migrates on CLI startup. No content storage — content lives in `kb.db`.
- **LLM**: Anthropic Claude via `llm.py` — centralized client with JSON parsing, Pydantic validation, and retry-with-backoff.
- **Schemas**: Pydantic v2 models in `schemas.py`. Hypothesis includes corpus fields: `supporting_insight_ids`, `contradicting_insight_ids`, `source_coverage`, `synthesis_note`.
- **Contracts**: `contracts.py` — domain registry loading and test_definition validation.
- **Stages**: `stages/orient.py`, `stages/retrieve.py`, `stages/distill.py`, `stages/translate.py`, `stages/migrate.py`.
- **Insight embedding**: `insight_embedder.py` — embeds insights to `insights_<domain>` chroma collection for semantic retrieval in Translate.
- **KB reader**: `kb_reader.py` — read-only access to `kb.db` and ChromaDB. No imports from knowledge-base package.
- **Prompts**: Plain text templates in `prompts/` with `{placeholder}` substitution.

## Data Ownership

- **knowledge-base owns content**: `content_item`, raw text, chunks, embeddings live in `kb.db` and chroma.
- **research-assistant owns research artifacts**: `domain_brief`, `insight`, `hypothesis`, provenance in `ra.db`.
- **Insights are dual-written**: `ra.db` (source of truth) + chroma `insights_<domain>` (searchable index for Translate).

## Pipeline Flow

1. `kb ingest url|file|batch` — ingest content into knowledge-base (run separately)
2. `ra orient` — build domain brief
3. `ra retrieve` — select kb content for distillation, writes retrieval_batch
4. `ra distill` — process pending batch items, extract insights, embed to chroma
5. `ra translate` — corpus-aware hypothesis generation via semantic retrieval over insights
6. `ra export` — Contract 1 JSON for factor-research
