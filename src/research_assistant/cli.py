import json
import sqlite3

import click

from research_assistant.config import Settings, get_settings, setup_logging
from research_assistant.db import get_connection, list_rows, migrate, resolve_domain
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
    migrate(conn)
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
@click.option("--url", required=True, help="Source URL")
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--trust-tier", type=click.Choice(["core", "supplementary", "exploratory"]), default="supplementary")
@click.option("--author", default=None, help="Source author (auto-detected if omitted)")
@pass_app
def ingest(app, url, domain, trust_tier, author):
    """Register and ingest a source."""
    from research_assistant.extractors.youtube import detect_source_type
    from research_assistant.stages.ingest import ingest_content, register_source

    source_type = detect_source_type(url)
    click.echo(f"Detected source type: {source_type}")

    sid = register_source(
        source_type=source_type,
        url=url,
        author=author or "Unknown",
        domain_id=domain,
        trust_tier=trust_tier,
        conn=app.conn,
    )

    click.echo(f"Ingesting content from {url}...")
    content = ingest_content(sid, app.conn, app.settings)

    status_color = {"success": "green", "partial": "yellow", "failed": "red"}
    click.echo(click.style(
        f"Status: {content.processing_status}", fg=status_color.get(content.processing_status, "white"),
    ))
    click.echo(f"Title: {content.title}")
    click.echo(f"Words: {content.word_count}")
    if content.error_detail:
        click.echo(click.style(f"Error: {content.error_detail}", fg="yellow"))


@cli.command("ingest-batch")
@click.option("--source-file", required=True, type=click.Path(exists=True), help="JSON file with source list")
@click.option("--domain", required=True, help="Domain ID or name")
@pass_app
def ingest_batch(app, source_file, domain):
    """Batch ingest from a source list file."""
    from research_assistant.stages.ingest import ingest_batch as _ingest_batch

    click.echo(f"Batch ingesting from {source_file}...")
    results = _ingest_batch(source_file, domain, app.conn, app.settings)
    success = sum(1 for r in results if r.processing_status == "success")
    click.echo(f"Ingested {len(results)} sources ({success} successful)")


@cli.command()
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--mode", type=click.Choice(["framework", "claim", "both"]), default="both")
@click.option("--content-id", default=None, help="Specific content ID to distill")
@click.option("--focus", default=None, help="Operator focus area")
@pass_app
def distill(app, domain, mode, content_id, focus):
    """Extract expert reasoning frameworks from ingested content."""
    from research_assistant.stages.distill import list_insights, run_distill, save_insights
    from research_assistant.stages.ingest import list_content

    if content_id:
        content_ids = [content_id]
    else:
        # Distill all content for the domain
        content_rows = list_content(domain, app.conn)
        content_ids = [r["content_id"] for r in content_rows]

    if not content_ids:
        click.echo("No content found to distill.")
        return

    total_insights = []
    for cid in content_ids:
        click.echo(f"Distilling content {cid}...")
        insights = run_distill(cid, domain, mode, focus, app.conn, app.settings)
        ids = save_insights(insights, app.conn)
        total_insights.extend(ids)
        click.echo(f"  Extracted {len(ids)} insights")

    click.echo(click.style(f"Total: {len(total_insights)} insights saved", fg="green"))


@cli.command()
@click.option("--domain", required=True, help="Domain ID or name")
@click.option("--mode", type=click.Choice(["explore", "commit"]), default="explore")
@click.option("--insight-id", multiple=True, help="Specific insight IDs to translate")
@click.option("--markets", default="", help="Comma-separated accessible markets")
@click.option("--data-sources", default="", help="Comma-separated available data sources")
@pass_app
def translate(app, domain, mode, insight_id, markets, data_sources):
    """Convert insights into testable hypothesis definitions."""
    from research_assistant.stages.distill import list_insights
    from research_assistant.stages.translate import (
        assess_feasibility,
        run_translate,
        save_hypotheses,
    )

    if insight_id:
        iids = list(insight_id)
    else:
        rows = list_insights(domain, app.conn, {"status": "active"})
        iids = [r["insight_id"] for r in rows]

    if not iids:
        click.echo("No insights found to translate.")
        return

    op_context = OperatorContext(
        accessible_markets=[m.strip() for m in markets.split(",") if m.strip()],
        available_data_sources=[d.strip() for d in data_sources.split(",") if d.strip()],
    )

    click.echo(f"Translating {len(iids)} insights in {mode} mode...")
    hypotheses = run_translate(iids, domain, mode, op_context, app.conn, app.settings)

    for h in hypotheses:
        assess_feasibility(h, op_context)

    ids = save_hypotheses(hypotheses, iids, app.conn)
    click.echo(click.style(f"Generated {len(ids)} hypotheses", fg="green"))
    for h in hypotheses:
        click.echo(f"  - {h.definition.name} (testability: {h.feasibility.estimated_testability})")


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

    sources = list_rows(app.conn, "source", {"domain_id": resolved})
    click.echo(f"Sources: {len(sources)}")
    for tier in ("core", "supplementary", "exploratory"):
        count = sum(1 for s in sources if s["trust_tier"] == tier)
        if count:
            click.echo(f"  {tier}: {count}")

    content = app.conn.execute(
        "SELECT processing_status, COUNT(*) as cnt FROM content_item ci "
        "JOIN source s ON ci.source_id = s.source_id "
        "WHERE s.domain_id = ? GROUP BY processing_status",
        (resolved,),
    ).fetchall()
    total_content = sum(r["cnt"] for r in content)
    click.echo(f"\nContent items: {total_content}")
    for r in content:
        click.echo(f"  {r['processing_status']}: {r['cnt']}")

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
        click.echo(f"{status_badge} {r['hypothesis_id'][:8]}... {defn.get('name', 'unnamed')}")
        click.echo(f"  Statement: {defn.get('statement', '')[:80]}")
        click.echo(f"  Testability: {feas.get('estimated_testability', 'unknown')}")
        click.echo()


@cli.command()
@click.option("--hypothesis-id", required=True, help="Hypothesis ID to export")
@click.option("--format", "fmt", type=click.Choice(["json"]), default="json")
@pass_app
def export(app, hypothesis_id, fmt):
    """Export hypothesis for testing harness."""
    from research_assistant.stages.translate import export_for_harness

    result = export_for_harness(hypothesis_id, app.conn)
    if not result:
        click.echo(click.style(f"Hypothesis not found: {hypothesis_id}", fg="red"))
        return

    click.echo(json.dumps(result, indent=2))
