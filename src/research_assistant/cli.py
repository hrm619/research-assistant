import json
import sqlite3

import click

from research_assistant.config import Settings, get_settings, setup_logging
from research_assistant.db import get_connection, list_rows, migrate as db_migrate, resolve_domain
from research_assistant.schemas import OperatorContext, OrientInput


pass_conn = click.make_pass_decorator(sqlite3.Connection, ensure=True)


class AppContext:
    def __init__(self, settings: Settings, conn: sqlite3.Connection):
        self.settings = settings
        self.conn = conn


pass_app = click.make_pass_decorator(AppContext)


@click.group()
@click.pass_context
def cli(ctx):
    """Research Assistant - domain research pipeline."""
    settings = get_settings()
    setup_logging(settings)
    conn = get_connection(settings.db_path)
    db_migrate(conn)
    ctx.obj = AppContext(settings, conn)


@cli.command()
@click.option("--domain", required=True, help="Domain name (e.g., 'fed_rate_decisions')")
@click.option("--market", required=True, help="Market type (e.g., 'kalshi', 'polymarket')")
@click.option("--known-domains", required=True, help="Comma-separated known domains")
@click.option("--seed-questions", default="", help="Comma-separated seed questions")
@click.option("--seed-sources", default="", help="Comma-separated seed source URLs")
@pass_app
def orient(app, domain, market, known_domains, seed_questions, seed_sources):
    """Build a structured understanding of a new domain."""
    from research_assistant.stages.orient import run_orient, save_domain_brief, validate_domain_brief

    input = OrientInput(
        domain_name=domain,
        market_type=market,
        operator_known_domains=[d.strip() for d in known_domains.split(",") if d.strip()],
        seed_questions=[q.strip() for q in seed_questions.split(",") if q.strip()],
        seed_sources=[s.strip() for s in seed_sources.split(",") if s.strip()],
    )

    click.echo(f"Orienting domain '{domain}' on {market}...")
    brief = run_orient(input, app.settings)

    errors = validate_domain_brief(brief)
    if errors:
        click.echo(click.style("Validation warnings:", fg="yellow"))
        for e in errors:
            click.echo(f"  - {e}")

    domain_id = save_domain_brief(brief, domain, market, app.conn)
    click.echo(click.style(f"Domain brief saved: {domain_id}", fg="green"))
    click.echo(f"Confidence: {brief.confidence}")
    click.echo(f"Analogies: {len(brief.analogies)}")
    click.echo(f"Open questions: {len(brief.open_questions)}")


@cli.command()
@click.option("--domain", required=True, help="Domain name (matches kb.db content_record.domain)")
@click.option("--trust-tier", default=None, help="Comma-separated trust tiers (core,supplementary,exploratory)")
@click.option("--analyst", default=None, help="Comma-separated analyst names")
@click.option("--source-type", default=None, help="Comma-separated source types (youtube,article,web,pdf)")
@click.option("--since", default=None, help="Published after (YYYY-MM-DD)")
@click.option("--until", "until_date", default=None, help="Published before (YYYY-MM-DD)")
@click.option("--limit", default=None, type=int, help="Max items to retrieve")
@click.option("--dry-run", is_flag=True, help="Preview matched content without writing")
@click.option("--force", is_flag=True, help="Include already-distilled items")
@pass_app
def retrieve(app, domain, trust_tier, analyst, source_type, since, until_date, limit, dry_run, force):
    """Select content from knowledge-base for distillation."""
    from research_assistant.kb_reader import get_kb_connection
    from research_assistant.stages.retrieve import run_retrieve

    kb_conn = get_kb_connection(app.settings.kb_db_path)

    trust_tiers = [t.strip() for t in trust_tier.split(",") if t.strip()] if trust_tier else None
    analysts = [a.strip() for a in analyst.split(",") if a.strip()] if analyst else None
    source_types = [s.strip() for s in source_type.split(",") if s.strip()] if source_type else None

    matched = run_retrieve(
        kb_conn=kb_conn,
        ra_conn=app.conn,
        domain=domain,
        trust_tiers=trust_tiers,
        analysts=analysts,
        source_types=source_types,
        since=since,
        until=until_date,
        limit=limit,
        force=force,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo(click.style(f"[dry-run] {len(matched)} content items matched:", fg="yellow"))
    else:
        click.echo(click.style(f"Retrieved {len(matched)} content items into batch:", fg="green"))

    for item in matched:
        tier_badge = click.style(f"[{item.get('trust_tier', '?')}]", fg="cyan")
        click.echo(f"  {tier_badge} {item['content_id'][:12]}... {item.get('analyst', '?')} — {item.get('title', 'untitled')}")

    if not matched:
        click.echo("No matching content found in kb.db.")


@cli.command()
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--mode", type=click.Choice(["framework", "claim", "both"]), default="both")
@click.option("--focus", default=None, help="Operator focus area")
@click.option("--from-kb", is_flag=True, help="Read all KB content directly (bypasses retrieval batch)")
@click.option("--kb-content-id", default=None, help="Specific KB content ID to distill directly")
@click.option("--kb-domain", default=None, help="KB collection name (defaults to --domain)")
@click.option("--batch-id", default=None, help="Specific retrieval batch ID to distill")
@click.option("--limit", "distill_limit", default=None, type=int, help="Max batch items to distill")
@pass_app
def distill(app, domain, mode, focus, from_kb, kb_content_id, kb_domain, batch_id, distill_limit):
    """Extract expert reasoning frameworks from content.

    Default: processes pending items from the most recent retrieval batch.
    Use --from-kb to read all KB content directly (bypasses batch).
    Use --kb-content-id for a specific KB content item.
    """
    from research_assistant.stages.distill import KBContext, run_distill, run_distill_batch, save_insights

    # Direct KB path: --from-kb or --kb-content-id
    if from_kb or kb_content_id:
        from research_assistant.kb_reader import get_chroma_client, get_kb_connection, list_kb_content

        collection_name = kb_domain or domain
        kb_conn = get_kb_connection(app.settings.kb_db_path)
        chroma_client = get_chroma_client(app.settings.chroma_persist_dir)
        kb_context = KBContext(kb_conn=kb_conn, chroma_client=chroma_client, collection_name=collection_name)

        if kb_content_id:
            content_ids = [kb_content_id]
        else:
            content_rows = list_kb_content(kb_conn, collection_name)
            content_ids = [r["content_id"] for r in content_rows if r.get("word_count", 0) > 0]
            click.echo(f"Found {len(content_ids)} content items in KB collection '{collection_name}'")

        if not content_ids:
            click.echo("No content found to distill.")
            return

        total_insights = []
        for cid in content_ids:
            click.echo(f"Distilling content {cid}...")
            try:
                insights = run_distill(cid, domain, mode, focus, app.conn, app.settings, kb_context=kb_context)
                ids = save_insights(insights, app.conn)
                total_insights.extend(ids)
                click.echo(f"  Extracted {len(ids)} insights")
            except Exception as e:
                click.echo(click.style(f"  Error: {e}", fg="red"))

        click.echo(click.style(f"Total: {len(total_insights)} insights saved", fg="green"))
        return

    # Batch-driven path: process pending retrieval_batch rows
    from research_assistant.kb_reader import get_chroma_client, get_kb_connection

    kb_conn = get_kb_connection(app.settings.kb_db_path)
    chroma_client = get_chroma_client(app.settings.chroma_persist_dir)

    openai_client = None
    if app.settings.openai_api_key:
        import openai
        openai_client = openai.OpenAI(api_key=app.settings.openai_api_key)

    click.echo(f"Distilling pending batch items for domain '{domain}'...")
    insights = run_distill_batch(
        domain=domain,
        mode=mode,
        focus=focus,
        conn=app.conn,
        settings=app.settings,
        kb_conn=kb_conn,
        chroma_client=chroma_client,
        batch_id=batch_id,
        limit=distill_limit,
        openai_client=openai_client,
    )

    if openai_client:
        from research_assistant.db import list_rows as _list_rows
        failed = _list_rows(app.conn, "insight_embedding", {"embedding_status": "failed"})
        if failed:
            click.echo(click.style(f"  {len(failed)} insights failed to embed (run 'ra reembed')", fg="yellow"))

    click.echo(click.style(f"Total: {len(insights)} insights extracted from batch", fg="green"))


@cli.command()
@click.option("--domain", required=True, help="Domain name")
@pass_app
def reembed(app, domain):
    """Retry failed or pending insight embeddings for a domain."""
    import openai as openai_mod
    from research_assistant.insight_embedder import reembed_failed
    from research_assistant.kb_reader import get_chroma_client

    if not app.settings.openai_api_key:
        click.echo(click.style("OPENAI_API_KEY required for embedding.", fg="red"))
        raise SystemExit(1)

    openai_client = openai_mod.OpenAI(api_key=app.settings.openai_api_key)
    chroma_client = get_chroma_client(app.settings.chroma_persist_dir)

    click.echo(f"Re-embedding failed/pending insights for domain '{domain}'...")
    success, failed = reembed_failed(domain, app.conn, chroma_client, openai_client, app.settings)
    click.echo(click.style(f"Embedded: {success}, Failed: {failed}", fg="green" if failed == 0 else "yellow"))


@cli.command()
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--mode", type=click.Choice(["explore", "commit"]), default="explore")
@click.option("--insight-id", multiple=True, help="Specific insight IDs to translate (legacy, non-corpus)")
@click.option("--seed-insight-id", default=None, help="Seed insight for corpus-aware translation")
@click.option("--corpus-k", default=15, type=int, help="Number of corpus insights to retrieve per seed")
@click.option("--include-exploratory", is_flag=True, help="Include exploratory trust tier in corpus retrieval")
@click.option("--markets", default="", help="Comma-separated accessible markets")
@click.option("--data-sources", default="", help="Comma-separated available data sources")
@click.option(
    "--domain-registry", default=None, type=click.Path(exists=True),
    help="Path to domain registry JSON for test_definition generation",
)
@pass_app
def translate(app, domain, mode, insight_id, seed_insight_id, corpus_k,
              include_exploratory, markets, data_sources, domain_registry):
    """Convert insights into testable hypothesis definitions.

    Default: corpus-aware translation. Selects seed insights and retrieves
    related insights from chroma for multi-source synthesis.

    Use --insight-id for legacy single-batch translation without corpus.
    Use --seed-insight-id to anchor on a specific insight.
    """
    from pathlib import Path

    from research_assistant.stages.translate import (
        assess_feasibility,
        run_translate,
        run_translate_corpus,
        save_hypotheses,
        select_seed_insights,
    )

    op_context = OperatorContext(
        accessible_markets=[m.strip() for m in markets.split(",") if m.strip()],
        available_data_sources=[d.strip() for d in data_sources.split(",") if d.strip()],
    )
    registry_path = Path(domain_registry) if domain_registry else None

    # Legacy path: --insight-id explicitly passed
    if insight_id:
        from research_assistant.stages.distill import list_insights

        iids = list(insight_id)
        click.echo(f"Translating {len(iids)} insights in {mode} mode (legacy)...")
        hypotheses = run_translate(
            iids, domain, mode, op_context, app.conn, app.settings,
            domain_registry_path=registry_path,
        )
        for h in hypotheses:
            assess_feasibility(h, op_context)
        ids = save_hypotheses(hypotheses, iids, app.conn)
        click.echo(click.style(f"Generated {len(ids)} hypotheses", fg="green"))
        for h in hypotheses:
            has_test_def = " [+test_definition]" if h.test_definition else ""
            click.echo(f"  - {h.definition.name} (testability: {h.feasibility.estimated_testability}){has_test_def}")
        return

    # Corpus-aware path
    from research_assistant.kb_reader import get_chroma_client
    chroma_client = get_chroma_client(app.settings.chroma_persist_dir)

    if seed_insight_id:
        seed_ids = [seed_insight_id]
    else:
        seed_ids = select_seed_insights(domain, app.conn)
        if not seed_ids:
            click.echo("No seed insights found to translate.")
            return

    click.echo(f"Translating {len(seed_ids)} seed insights in {mode} mode (corpus-aware)...")
    if registry_path:
        click.echo(f"Using domain registry: {registry_path}")

    all_hypotheses = []
    for sid in seed_ids:
        click.echo(f"  Seed: {sid[:12]}...")
        try:
            hypotheses = run_translate_corpus(
                seed_insight_id=sid,
                domain_id=domain,
                mode=mode,
                operator_context=op_context,
                conn=app.conn,
                settings=app.settings,
                chroma_client=chroma_client,
                corpus_k=corpus_k,
                include_exploratory=include_exploratory,
                domain_registry_path=registry_path,
            )
            for h in hypotheses:
                assess_feasibility(h, op_context)
            ids = save_hypotheses(hypotheses, [sid], app.conn)
            all_hypotheses.extend(hypotheses)
            for h in hypotheses:
                coverage = f" [{h.source_coverage.n_sources} sources]" if h.source_coverage else ""
                click.echo(f"    → {h.definition.name}{coverage}")
        except Exception as e:
            click.echo(click.style(f"    Error: {e}", fg="red"))

    click.echo(click.style(f"Total: {len(all_hypotheses)} hypotheses generated", fg="green"))


@cli.group()
def show():
    """Show detailed entity data."""
    pass


@show.command("domain")
@click.option("--domain", required=True, help="Domain ID or name")
@pass_app
def show_domain(app, domain):
    """Pretty-print the full domain brief."""
    from research_assistant.db import get_row

    resolved = resolve_domain(app.conn, domain)
    if not resolved:
        click.echo(click.style(f"Domain not found: {domain}", fg="red"))
        return

    row = get_row(app.conn, "domain_brief", "domain_id", resolved)
    brief = json.loads(row["brief_json"])

    click.echo(click.style(f"Domain: {row['domain_name']}", bold=True))
    click.echo(f"Market: {row['market_type']}  |  Status: {row['status']}  |  Confidence: {brief.get('confidence', '?')}")
    click.echo(f"Created: {row['created_at']}")
    click.echo()

    mm = brief.get("market_mechanics", {})
    click.echo(click.style("Market Mechanics", bold=True))
    click.echo(f"  Instrument: {mm.get('instrument_type', '?')}")
    click.echo(f"  Settlement: {mm.get('settlement', '?')}")
    click.echo(f"  Liquidity: {mm.get('liquidity_profile', '?')}")
    click.echo(f"  Fees: {mm.get('fee_structure', '?')}")
    click.echo(f"  Positions: {', '.join(mm.get('position_types', []))}")
    if mm.get("known_biases"):
        click.echo(f"  Known biases: {', '.join(mm['known_biases'])}")
    click.echo()

    gt = brief.get("game_theory", {})
    click.echo(click.style("Game Theory", bold=True))
    click.echo(f"  Participants: {', '.join(gt.get('participant_types', []))}")
    for asym in gt.get("information_asymmetries", []):
        click.echo(f"  Info asymmetry: {asym}")
    for mistake in gt.get("common_mistakes", []):
        click.echo(f"  Common mistake: {mistake}")
    click.echo()

    meta = brief.get("current_meta", {})
    click.echo(click.style("Current Meta", bold=True))
    click.echo(f"  Consensus: {meta.get('consensus_view', '?')}")
    for narr in meta.get("dominant_narratives", []):
        click.echo(f"  Narrative: {narr}")
    for angle in meta.get("contrarian_angles", []):
        click.echo(f"  Contrarian: {angle}")
    click.echo()

    analogies = brief.get("analogies", [])
    click.echo(click.style(f"Analogies ({len(analogies)})", bold=True))
    for a in analogies:
        click.echo(f"  {a.get('known_domain', '?')} -> {a.get('mapping', '?')}")
        click.echo(click.style(f"    Breaks: {a.get('where_analogy_breaks', '?')}", fg="yellow"))
    click.echo()

    sources = brief.get("key_data_sources", [])
    if sources:
        click.echo(click.style("Key Data Sources", bold=True))
        for s in sources:
            click.echo(f"  - {s}")
        click.echo()

    questions = brief.get("open_questions", [])
    click.echo(click.style(f"Open Questions ({len(questions)})", bold=True))
    for q in questions:
        click.echo(f"  ? {q}")


@show.command("insight")
@click.option("--id", "insight_id", required=True, help="Insight ID")
@pass_app
def show_insight(app, insight_id):
    """Pretty-print a single insight."""
    from research_assistant.db import get_row

    row = get_row(app.conn, "insight", "insight_id", insight_id)
    if not row:
        click.echo(click.style(f"Insight not found: {insight_id}", fg="red"))
        return

    click.echo(click.style(f"[{row['insight_type']}]", fg="cyan", bold=True) + f"  {row['insight_id']}")
    click.echo(f"Content: {row['content_id']}  |  Status: {row['status']}")
    click.echo(f"Extracted: {row['extracted_at']}")
    click.echo(f"Source ref: {row['source_quote_ref']}")
    click.echo()

    if row.get("framework_json"):
        fw = json.loads(row["framework_json"])
        click.echo(click.style("Framework", bold=True))
        click.echo(f"  Name: {fw.get('name', '?')}")
        click.echo(f"  Attribution: {fw.get('author_attribution', '?')}")
        click.echo(f"  Mechanism: {fw.get('mechanism', '?')}")
        if fw.get("conditions"):
            click.echo("  Conditions:")
            for c in fw["conditions"]:
                click.echo(f"    - {c}")
        if fw.get("predictions"):
            click.echo("  Predictions:")
            for p in fw["predictions"]:
                click.echo(f"    - {p}")
        if fw.get("assumptions"):
            click.echo("  Assumptions:")
            for a in fw["assumptions"]:
                click.echo(f"    - {a}")
        if fw.get("evidence_cited"):
            click.echo("  Evidence:")
            for e in fw["evidence_cited"]:
                click.echo(f"    - {e}")
        click.echo(f"  Confidence language: {fw.get('confidence_language', '?')}")

    if row.get("claim_json"):
        cl = json.loads(row["claim_json"])
        click.echo(click.style("Claim", bold=True))
        click.echo(f"  Statement: {cl.get('statement', '?')}")
        click.echo(f"  Reasoning: {cl.get('reasoning', '?')}")
        if cl.get("timeframe"):
            click.echo(f"  Timeframe: {cl['timeframe']}")
        click.echo(f"  Falsifiable: {cl.get('falsifiable', '?')}")
        if cl.get("falsification_trigger"):
            click.echo(f"  Trigger: {cl['falsification_trigger']}")

    if row.get("operator_note"):
        click.echo()
        click.echo(click.style("Operator note:", bold=True) + f" {row['operator_note']}")


@show.command("hypothesis")
@click.option("--id", "hypothesis_id", required=True, help="Hypothesis ID")
@pass_app
def show_hypothesis(app, hypothesis_id):
    """Pretty-print a single hypothesis."""
    from research_assistant.db import get_row

    row = get_row(app.conn, "hypothesis", "hypothesis_id", hypothesis_id)
    if not row:
        click.echo(click.style(f"Hypothesis not found: {hypothesis_id}", fg="red"))
        return

    defn = json.loads(row["definition_json"])
    feas = json.loads(row["feasibility_json"])
    chain = json.loads(row["reasoning_chain_json"])

    status_colors = {"draft": "yellow", "accepted": "green", "rejected": "red", "tested": "cyan"}
    click.echo(
        click.style(f"[{row['status']}]", fg=status_colors.get(row["status"], "white"), bold=True)
        + f"  {row['hypothesis_id']}"
    )
    click.echo(f"Created: {row['created_at']}")
    click.echo()

    click.echo(click.style("Definition", bold=True))
    click.echo(f"  Name: {defn.get('name', '?')}")
    click.echo(f"  Statement: {defn.get('statement', '?')}")
    click.echo(f"  Factor: {defn.get('factor', '?')}")
    click.echo(f"  Classification: {defn.get('classification', '?')}")
    click.echo(f"  Outcome: {defn.get('outcome_measure', '?')}")
    click.echo(f"  Timeframe: {defn.get('timeframe', '?')}")
    click.echo(f"  Data available: {defn.get('data_available', '?')}")
    click.echo(f"  Market expression: {defn.get('market_expression', '?')}")
    if defn.get("data_required"):
        click.echo("  Data required:")
        for d in defn["data_required"]:
            click.echo(f"    - {d}")
    click.echo()

    click.echo(click.style("Feasibility", bold=True))
    click.echo(f"  Testability: {feas.get('estimated_testability', '?')}")
    if feas.get("minimum_sample_size"):
        click.echo(f"  Min sample size: {feas['minimum_sample_size']}")
    if feas.get("data_gap"):
        click.echo("  Data gaps:")
        for g in feas["data_gap"]:
            click.echo(f"    - {g}")
    if feas.get("knowledge_gap"):
        click.echo("  Knowledge gaps:")
        for g in feas["knowledge_gap"]:
            click.echo(f"    - {g}")
    click.echo()

    click.echo(click.style("Reasoning Chain", bold=True))
    click.echo(f"  From insight: {chain.get('from_insight', '?')}")
    click.echo(f"  Logic: {chain.get('translation_logic', '?')}")
    if chain.get("assumptions_added"):
        click.echo("  Assumptions added:")
        for a in chain["assumptions_added"]:
            click.echo(click.style(f"    - {a}", fg="yellow"))
    if chain.get("weaknesses"):
        click.echo("  Weaknesses:")
        for w in chain["weaknesses"]:
            click.echo(click.style(f"    - {w}", fg="red"))

    if row.get("operator_note"):
        click.echo()
        click.echo(click.style("Operator note:", bold=True) + f" {row['operator_note']}")


@cli.command()
@click.option("--domain", required=True, help="Domain ID or name")
@pass_app
def status(app, domain):
    """Review pipeline state for a domain."""
    resolved = resolve_domain(app.conn, domain)
    if not resolved:
        click.echo(click.style(f"Domain not found: {domain}", fg="red"))
        return

    brief = app.conn.execute(
        "SELECT * FROM domain_brief WHERE domain_id = ?", (resolved,)
    ).fetchone()
    click.echo(f"Domain: {brief['domain_name']} ({brief['market_type']})")
    click.echo(f"Status: {brief['status']}")
    click.echo()

    domain_name = brief["domain_name"]
    batch_rows = app.conn.execute(
        "SELECT distill_status, COUNT(*) as cnt FROM retrieval_batch "
        "WHERE domain = ? GROUP BY distill_status",
        (domain_name,),
    ).fetchall()
    total_batch = sum(r["cnt"] for r in batch_rows)
    if total_batch:
        click.echo(f"\nRetrieval batch (from kb): {total_batch}")
        for r in batch_rows:
            click.echo(f"  {r['distill_status']}: {r['cnt']}")

    insights = list_rows(app.conn, "insight", {"domain_id": resolved})
    click.echo(f"\nInsights: {len(insights)}")
    for itype in ("framework", "claim", "observation"):
        count = sum(1 for i in insights if i["insight_type"] == itype)
        if count:
            click.echo(f"  {itype}: {count}")

    hypotheses = list_rows(app.conn, "hypothesis", {"domain_id": resolved})
    click.echo(f"\nHypotheses: {len(hypotheses)}")
    for hstatus in ("draft", "review", "accepted", "rejected", "tested"):
        count = sum(1 for h in hypotheses if h["status"] == hstatus)
        if count:
            click.echo(f"  {hstatus}: {count}")


@cli.group("list")
def list_cmd():
    """List entities."""
    pass


@list_cmd.command("insights")
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--type", "insight_type", default=None, help="Filter by type: framework, claim, observation")
@click.option("--status", "insight_status", default=None, help="Filter by status: active, merged, discarded")
@pass_app
def list_insights_cmd(app, domain, insight_type, insight_status):
    """List insights for a domain."""
    from research_assistant.stages.distill import list_insights

    filters = {}
    if insight_type:
        filters["insight_type"] = insight_type
    if insight_status:
        filters["status"] = insight_status

    rows = list_insights(domain, app.conn, filters if filters else None)
    if not rows:
        click.echo("No insights found.")
        return

    for r in rows:
        type_badge = click.style(f"[{r['insight_type']}]", fg="cyan")
        click.echo(f"{type_badge} {r['insight_id'][:8]}...")
        if r.get("framework_json"):
            fw = json.loads(r["framework_json"])
            click.echo(f"  Name: {fw.get('name', 'unnamed')}")
            click.echo(f"  Mechanism: {fw.get('mechanism', '')[:80]}...")
        if r.get("claim_json"):
            cl = json.loads(r["claim_json"])
            click.echo(f"  Statement: {cl.get('statement', '')[:80]}...")
        click.echo()


@list_cmd.command("hypotheses")
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--status", "hyp_status", default=None, help="Filter by status")
@pass_app
def list_hypotheses_cmd(app, domain, hyp_status):
    """List hypotheses for a domain."""
    from research_assistant.stages.translate import list_hypotheses

    filters = {}
    if hyp_status:
        filters["status"] = hyp_status

    rows = list_hypotheses(domain, app.conn, filters if filters else None)
    if not rows:
        click.echo("No hypotheses found.")
        return

    for r in rows:
        defn = json.loads(r["definition_json"])
        feas = json.loads(r["feasibility_json"])
        status_colors = {"draft": "yellow", "accepted": "green", "rejected": "red"}
        status_badge = click.style(f"[{r['status']}]", fg=status_colors.get(r["status"], "white"))
        click.echo(f"{status_badge} {r['hypothesis_id']} {defn.get('name', 'unnamed')}")
        click.echo(f"  Statement: {defn.get('statement', '')[:80]}")
        click.echo(f"  Testability: {feas.get('estimated_testability', 'unknown')}")
        click.echo()


@cli.command()
@click.option("--hypothesis-id", required=True, help="Hypothesis ID to export")
@click.option("--format", "fmt", type=click.Choice(["json"]), default="json")
@click.option(
    "--domain-registry", default=None, type=click.Path(exists=True),
    help="Path to domain registry JSON for validation",
)
@click.option(
    "--output", "-o", default=None, type=click.Path(),
    help="Output file path (default: stdout)",
)
@pass_app
def export(app, hypothesis_id, fmt, domain_registry, output):
    """Export hypothesis as Contract 1 JSON for factor-research."""
    from pathlib import Path

    from research_assistant.stages.translate import export_for_harness

    registry = None
    if domain_registry:
        from research_assistant.contracts import load_domain_registry
        registry = load_domain_registry(Path(domain_registry))

    output_path = Path(output) if output else None

    try:
        result = export_for_harness(
            hypothesis_id, app.conn,
            domain_registry=registry,
            output_path=output_path,
        )
    except ValueError as e:
        click.echo(click.style(str(e), fg="red"), err=True)
        raise SystemExit(1)

    if not result:
        click.echo(click.style(f"Hypothesis not found: {hypothesis_id}", fg="red"))
        raise SystemExit(1)

    if output_path:
        click.echo(click.style(f"Contract 1 written to {output_path}", fg="green"))
    else:
        click.echo(json.dumps(result, indent=2))


@cli.group()
def migrate():
    """Database migration commands."""
    pass


@migrate.command("to-kb-ownership")
@click.option("--dry-run", is_flag=True, help="Preview changes without modifying the database")
@pass_app
def migrate_to_kb(app, dry_run):
    """Migrate ra.db to kb-ownership model.

    Backs up ra.db, remaps insight references, drops content_item and source tables.
    Run with --dry-run first to preview changes.
    """
    from research_assistant.stages.migrate import run_migration

    kb_conn = None
    try:
        from research_assistant.kb_reader import get_kb_connection
        kb_conn = get_kb_connection(app.settings.kb_db_path)
    except FileNotFoundError:
        click.echo(click.style("Warning: kb.db not found. Content matching will be skipped.", fg="yellow"))

    report = run_migration(app.conn, kb_conn, app.settings.db_path, dry_run=dry_run)

    if dry_run:
        click.echo(click.style("[dry-run] Migration preview:", fg="yellow"))
    else:
        click.echo(click.style("Migration complete.", fg="green"))
        if report["backup_path"]:
            click.echo(f"Backup: {report['backup_path']}")

    unmatched = report["unmatched_content"]
    if unmatched:
        click.echo(f"\nUnmatched content items ({len(unmatched)}):")
        click.echo("  These exist in ra.db but not in kb.db. Re-ingest via 'kb ingest'.")
        for item in unmatched[:10]:
            click.echo(f"  - {item['content_id'][:12]}... {item.get('title', 'untitled')}")
        if len(unmatched) > 10:
            click.echo(f"  ... and {len(unmatched) - 10} more")
    else:
        click.echo("\nAll content matched or no content_item table found.")

    if not dry_run:
        stats = report["remap_stats"]
        click.echo(f"\nInsight remapping: {stats.get('remapped', 0)} remapped, "
                    f"{stats.get('already_set', 0)} already set, "
                    f"{stats.get('orphaned', 0)} orphaned")
        click.echo(f"Dropped tables: {', '.join(report['dropped_tables']) or 'none'}")

    if not dry_run:
        click.echo(f"\nRollback: cp {report.get('backup_path', '<backup>')} {app.settings.db_path}")
