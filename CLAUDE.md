# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research assistant that accelerates progression from domain curiosity to testable hypothesis. Four-stage pipeline: **Orient** (domain mapping) → **Ingest** (content extraction) → **Distill** (reasoning extraction) → **Translate** (hypothesis generation). Build spec lives in `research_assistant_build_spec.docx`.

## Development Setup

Uses **uv** for package management (not pip). Python 3.13.

```bash
uv sync          # install dependencies
uv run ra --help # CLI entry point
```

Requires `ANTHROPIC_API_KEY` in `.env` (gitignored) for LLM calls. Config in `src/research_assistant/config.py` — defaults DB to `~/.research-assistant/ra.db`.

## Testing

```bash
uv run pytest                        # run all 77 tests
uv run pytest tests/test_foo.py      # single file
uv run pytest -k "test_name"         # single test by name
```

All LLM calls and YouTube extraction are mocked in tests. No API keys needed to run tests.

## CLI Commands

### Pipeline stages
```bash
ra orient --domain "fed_rate_decisions" --market kalshi --known-domains "sports,politics"
ra ingest --url "https://youtube.com/watch?v=..." --domain fed_rate_decisions --trust-tier core
ra ingest-batch --source-file sources.json --domain fed_rate_decisions
ra distill --domain fed_rate_decisions --mode both
ra translate --domain fed_rate_decisions --mode explore
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
```

## Architecture

- **Package layout**: `src/research_assistant/` (src layout, hatchling build)
- **Entry point**: `ra` → `research_assistant.cli:cli` (Click)
- **DB**: SQLite via `db.py` — all entities stored with JSON blob columns for nested data. Schema auto-migrates on CLI startup.
- **LLM**: Anthropic Claude (`claude-sonnet-4-20250514`) via `llm.py` — centralized client with JSON parsing, Pydantic validation, and retry-with-backoff. Re-prompts on validation failure. Model configurable via `LLM_MODEL` env var.
- **Schemas**: Pydantic v2 models in `schemas.py` for all entities with cross-field validators (e.g., framework insights must have mechanism, hypotheses must have weaknesses)
- **Stages**: Each in `stages/` — `orient.py`, `ingest.py`, `distill.py`, `translate.py`. Each follows the pattern: build prompt → call LLM → validate → save to DB.
- **Extractors**: `extractors/youtube.py` (MVP). Uses yt-dlp for metadata + subtitle extraction.
- **Prompts**: Plain text templates in `prompts/` with `{placeholder}` substitution, loaded via `importlib.resources`.

## MVP Scope

Only YouTube ingestion is implemented. Other extractors (Substack, web article, PDF, email) are deferred per the build spec. Dedup uses naive name matching; embedding-based dedup is deferred.
