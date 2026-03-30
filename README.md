# Research Assistant

Research assistant that accelerates progression from domain curiosity to testable hypothesis. Four-stage pipeline:

**Orient** (domain mapping) &rarr; **Ingest** (content extraction) &rarr; **Distill** (reasoning extraction) &rarr; **Translate** (hypothesis generation)

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### 1. Orient -- map a new domain

```bash
ra orient --domain "fed_rate_decisions" --market kalshi --known-domains "sports,politics"
```

Builds a structured domain brief: market mechanics, game theory, current meta, analogies, and open questions.

### 2. Ingest -- extract content from sources

```bash
ra ingest --url "https://youtube.com/watch?v=..." --domain fed_rate_decisions --trust-tier core
ra ingest-batch --source-file sources.json --domain fed_rate_decisions
```

Currently supports YouTube videos (metadata + transcript via yt-dlp). Other source types (Substack, PDF, web articles) are planned.

### 3. Distill -- extract expert reasoning

```bash
ra distill --domain fed_rate_decisions --mode both
```

Extracts frameworks (mental models with mechanisms, conditions, predictions) and claims (falsifiable statements with timeframes) from ingested content.

### 4. Translate -- generate testable hypotheses

```bash
ra translate --domain fed_rate_decisions --mode explore
```

Converts insights into structured hypothesis definitions with feasibility assessments, data requirements, and market expressions.

### Inspection & export

```bash
ra status --domain fed_rate_decisions          # pipeline state overview
ra show domain --domain fed_rate_decisions     # full domain brief
ra show insight --id <insight-id>              # detailed insight view
ra show hypothesis --id <hypothesis-id>        # detailed hypothesis view
ra list insights --domain fed_rate_decisions   # list insights (filterable by --type, --status)
ra list hypotheses --domain fed_rate_decisions # list hypotheses (filterable by --status)
ra export --hypothesis-id <id> --format json   # export for testing harness
```

## Testing

```bash
uv run pytest              # all tests
uv run pytest -k "test_x"  # single test
```

All LLM calls and YouTube extraction are mocked. No API keys needed.

## Architecture

```
src/research_assistant/
  cli.py          # Click CLI entry point
  config.py       # Settings (pydantic-settings, .env)
  db.py           # SQLite with JSON blob columns, auto-migration
  llm.py          # Anthropic client with retries & Pydantic validation
  schemas.py      # Pydantic v2 models for all entities
  stages/         # orient, ingest, distill, translate
  extractors/     # youtube (MVP)
  prompts/        # plain text templates with {placeholder} substitution
```
