# Migration: ra.db to kb-ownership

This document covers migrating from the old research-assistant pipeline (with `ra ingest`) to the new pipeline where `knowledge-base` owns all content.

## What changed

- `ra ingest` and `ra ingest-batch` commands removed
- `content_item` and `source` tables dropped from `ra.db`
- Content ingestion is now handled exclusively by `kb ingest`
- Distill reads content from `kb.db` via `retrieval_batch`
- Translate uses semantic retrieval over embedded insights for corpus-aware hypothesis generation

## Migration steps

### 1. Ensure kb.db has your content

```bash
kb backfill [--domain NAME]
kb status [--domain NAME]
```

### 2. Preview the migration

```bash
ra migrate to-kb-ownership --dry-run
```

This shows:
- Content items in `ra.db` that don't exist in `kb.db` (need re-ingestion)
- How many insights will be remapped

### 3. Re-ingest unmatched content (if any)

If the dry-run shows unmatched content, re-ingest it via kb:

```bash
kb ingest url <URL> --analyst <NAME> --trust-tier <TIER> --domain <DOMAIN>
```

### 4. Run the migration

```bash
ra migrate to-kb-ownership
```

This will:
1. Back up `ra.db` to `ra.db.bak-<timestamp>`
2. Remap `insight.content_item_ref` to point to `kb.db` content IDs
3. Drop `content_item` and `source` tables
4. Report results

### 5. Verify

```bash
ra status --domain <each_domain>
```

Insights should be intact. Hypothesis counts unchanged.

## Rollback

```bash
cp ~/.research-assistant/ra.db.bak-<timestamp> ~/.research-assistant/ra.db
```

## New workflow

```bash
# 1. Ingest content via knowledge-base
kb ingest url "https://youtube.com/..." --analyst barrett --trust-tier core --domain nfl

# 2. Select content for distillation
ra retrieve --domain nfl --trust-tier core,supplementary

# 3. Distill (processes pending batch items)
ra distill --domain nfl

# 4. Generate corpus-aware hypotheses
ra translate --domain nfl --mode explore

# 5. Export for factor-research
ra export --hypothesis-id <id> --domain-registry <path> --output <path>
```
