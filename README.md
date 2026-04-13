# Research Assistant

Research assistant that accelerates progression from domain curiosity to testable hypothesis. Four-stage pipeline:

**Orient** (domain mapping) &rarr; **Retrieve** (select kb content) &rarr; **Distill** (reasoning extraction) &rarr; **Translate** (corpus-aware hypothesis generation)

Content ingestion is handled by the [knowledge-base](https://github.com/hrm619/knowledge-base) repo. This repo consumes `kb.db` read-only.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...                           # for insight embedding (optional)
KB_DB_PATH=~/.knowledge-base/kb.db              # default
CHROMA_PERSIST_DIR=~/.knowledge-base/chroma     # default
```

## Usage

### 1. Orient -- map a new domain

```bash
ra orient --domain "fed_rate_decisions" --market kalshi --known-domains "sports,politics"
```

Builds a structured domain brief: market mechanics, game theory, current meta, analogies, and open questions.

### 2. Retrieve -- select content from knowledge-base

```bash
ra retrieve --domain nfl --trust-tier core,supplementary --since 2025-01-01
ra retrieve --domain nfl --analyst barrett,jj --dry-run
```

Queries `kb.db` for content matching the filters and creates a retrieval batch for distillation. Use `--dry-run` to preview without writing.

### 3. Distill -- extract expert reasoning

```bash
ra distill --domain nfl                          # process pending batch items
ra distill --domain nfl --from-kb                # direct KB path (bypasses batch)
ra distill --domain nfl --batch-id <uuid>        # specific batch
```

Extracts frameworks (mental models with mechanisms, conditions, predictions) and claims (falsifiable statements with timeframes). When `OPENAI_API_KEY` is set, insights are automatically embedded to chroma for corpus retrieval in Translate.

### 4. Translate -- generate corpus-aware hypotheses

```bash
ra translate --domain nfl --mode explore                    # walks seed insights with corpus synthesis
ra translate --seed-insight-id <id> --domain nfl            # anchor on specific insight
ra translate --domain nfl --domain-registry <path>          # with metrics catalog constraint
ra translate --insight-id <id> --insight-id <id2> --domain nfl  # legacy non-corpus path
```

Performs semantic retrieval over embedded insights to find corroborating and contradicting evidence from other experts. Hypotheses include `supporting_insight_ids`, `contradicting_insight_ids`, `source_coverage`, and `synthesis_note`.

### Utility commands

```bash
ra reembed --domain nfl                          # retry failed insight embeddings
ra migrate to-kb-ownership --dry-run             # preview migration from old schema
ra migrate to-kb-ownership                       # execute migration
```

### Inspection & export

```bash
ra status --domain fed_rate_decisions          # pipeline state overview
ra show domain --domain fed_rate_decisions     # full domain brief
ra show insight --id <insight-id>              # detailed insight view
ra show hypothesis --id <hypothesis-id>        # detailed hypothesis view
ra list insights --domain fed_rate_decisions   # list insights (filterable by --type, --status)
ra list hypotheses --domain fed_rate_decisions # list hypotheses (filterable by --status)
ra export --hypothesis-id <id> --format json   # export to stdout
ra export --hypothesis-id <id> --domain-registry <path> --output <path>  # Contract 1 JSON
```

## Testing

```bash
uv run pytest              # all 165 tests
uv run pytest -k "test_x"  # single test
```

All LLM, OpenAI, and ChromaDB calls are mocked. No API keys needed.

## Architecture

```
src/research_assistant/
  cli.py              # Click CLI entry point
  config.py           # Settings (pydantic-settings, .env)
  db.py               # SQLite with JSON blob columns, auto-migration
  llm.py              # Anthropic client with retries & Pydantic validation
  schemas.py          # Pydantic v2 models (Insight, Hypothesis w/ corpus fields, etc.)
  contracts.py        # Domain registry loading & test_definition validation
  kb_reader.py        # Read-only access to kb.db and ChromaDB
  insight_embedder.py # Embed insights to chroma for semantic retrieval
  stages/
    orient.py         # Domain mapping via LLM
    retrieve.py       # Select kb.db content for distillation
    distill.py        # Extract frameworks and claims via LLM
    translate.py      # Corpus-aware hypothesis generation
    migrate.py        # Migration from old schema
  prompts/            # Plain text templates with {placeholder} substitution
```

## Migration from pre-refactor

If you have an existing `ra.db` with `content_item` and `source` tables, see [MIGRATION.md](MIGRATION.md).
