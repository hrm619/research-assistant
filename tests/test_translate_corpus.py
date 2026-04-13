"""Tests for corpus-aware translate: semantic retrieval + synthesis."""

import json
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from research_assistant.config import Settings
from research_assistant.db import get_connection, get_row, insert_row, list_rows, migrate
from research_assistant.schemas import Hypothesis, OperatorContext, SourceCoverage
from research_assistant.stages.translate import (
    _build_corpus_prompt_section,
    _summarize_insight,
    build_corpus_translate_prompt,
    query_corpus,
    run_translate_corpus,
    save_hypotheses,
    select_seed_insights,
)


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key", llm_max_retries=1, llm_backoff_base=0.01)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    migrate(c)
    insert_row(c, "domain_brief", {
        "domain_id": "d1",
        "domain_name": "nfl",
        "market_type": "polymarket",
        "created_at": "2026-01-01T00:00:00Z",
        "brief_json": '{"confidence": "medium"}',
        "status": "active",
    })
    return c


def _insert_insight(conn, insight_id, analyst="barrett", trust_tier="core",
                    insight_type="framework", fw_name="test_fw"):
    conn.execute("PRAGMA foreign_keys=OFF")
    data = {
        "insight_id": insight_id,
        "content_id": f"c-{insight_id}",
        "content_item_ref": f"c-{insight_id}",
        "source_id": "",
        "domain_id": "d1",
        "extracted_at": "2026-01-01T00:00:00Z",
        "insight_type": insight_type,
        "framework_json": json.dumps({
            "name": fw_name,
            "author_attribution": analyst,
            "mechanism": f"Mechanism for {fw_name}",
            "conditions": ["condition A"],
            "predictions": ["prediction X"],
            "assumptions": ["assumption 1"],
            "evidence_cited": ["data"],
            "confidence_language": "strong",
        }) if insight_type == "framework" else None,
        "claim_json": json.dumps({
            "statement": f"Claim by {analyst}",
            "reasoning": "Because reasons",
            "timeframe": "2025",
            "falsifiable": True,
            "falsification_trigger": "If wrong",
        }) if insight_type == "claim" else None,
        "source_quote_ref": "test",
        "status": "active",
        "analyst": analyst,
        "trust_tier": trust_tier,
        "content_source": "kb",
    }
    insert_row(conn, "insight", data)
    conn.execute("PRAGMA foreign_keys=ON")
    return data


@pytest.fixture
def chroma_client():
    client = chromadb.Client()
    for coll in client.list_collections():
        client.delete_collection(coll.name)
    return client


def _populate_insights_chroma(chroma_client, insights_data, domain="nfl"):
    coll = chroma_client.get_or_create_collection(
        f"insights_{domain}", metadata={"hnsw:space": "cosine"},
    )
    ids = []
    docs = []
    metas = []
    for d in insights_data:
        ids.append(d["insight_id"])
        text = f"[TYPE: {d['insight_type']}] [ANALYST: {d.get('analyst','')}]\n"
        if d.get("framework_json"):
            fw = json.loads(d["framework_json"])
            text += f"{fw['name']}: {fw['mechanism']}"
        elif d.get("claim_json"):
            cl = json.loads(d["claim_json"])
            text += cl["statement"]
        docs.append(text)
        metas.append({
            "insight_id": d["insight_id"],
            "insight_type": d["insight_type"],
            "analyst": d.get("analyst", ""),
            "trust_tier": d.get("trust_tier", ""),
            "domain": domain,
            "content_item_ref": d.get("content_item_ref", ""),
            "status": d.get("status", "active"),
            "content_source": d.get("content_source", "kb"),
        })
    coll.add(ids=ids, documents=docs, metadatas=metas)
    return coll


SAMPLE_CORPUS_HYPOTHESIS = {
    "definition": {
        "name": "snap_share_rb_value",
        "statement": "RBs with 70%+ snap share finish top 12",
        "factor": "snap_share",
        "classification": "binary",
        "outcome_measure": "Top 12 RB finish",
        "timeframe": "season",
        "data_required": ["snap counts"],
        "data_available": True,
        "market_expression": "Draft RBs with projected 70%+ snap share",
    },
    "feasibility": {
        "data_gap": [],
        "knowledge_gap": ["sample size across seasons"],
        "minimum_sample_size": 30,
        "estimated_testability": "high",
    },
    "reasoning_chain": {
        "from_insight": "Barrett's workload value framework",
        "translation_logic": "High snap share -> volume -> fantasy points -> top finish",
        "assumptions_added": ["Snap share stable week to week"],
        "weaknesses": ["Injury risk not accounted for"],
    },
    "supporting_insight_ids": ["i2"],
    "contradicting_insight_ids": ["i3"],
    "source_coverage": {"analysts": ["barrett", "jj", "winks"], "trust_tiers": ["core", "supplementary"], "n_sources": 3},
    "synthesis_note": "Barrett and JJ agree on volume; Winks contradicts on efficiency mattering more than volume.",
}


class TestQueryCorpus:
    def test_returns_hydrated_insights(self, conn, chroma_client):
        seed = _insert_insight(conn, "i1", analyst="barrett", fw_name="workload_signal")
        i2 = _insert_insight(conn, "i2", analyst="jj", fw_name="volume_signal")
        i3 = _insert_insight(conn, "i3", analyst="winks", fw_name="efficiency_counter")
        _populate_insights_chroma(chroma_client, [seed, i2, i3])

        results = query_corpus(seed, "nfl", conn, chroma_client, n_results=5)
        assert len(results) >= 1
        ids = {r["insight_id"] for r in results}
        assert "i1" not in ids  # seed excluded

    def test_excludes_seed_insight(self, conn, chroma_client):
        seed = _insert_insight(conn, "i1")
        _populate_insights_chroma(chroma_client, [seed])

        results = query_corpus(seed, "nfl", conn, chroma_client, n_results=5)
        assert len(results) == 0

    def test_empty_collection_returns_empty(self, conn, chroma_client):
        seed = _insert_insight(conn, "i1")
        results = query_corpus(seed, "nfl", conn, chroma_client, n_results=5)
        assert results == []

    def test_trust_tier_filter(self, conn, chroma_client):
        seed = _insert_insight(conn, "i1", trust_tier="core")
        i2 = _insert_insight(conn, "i2", trust_tier="core", fw_name="core_fw")
        i3 = _insert_insight(conn, "i3", trust_tier="exploratory", fw_name="exp_fw")
        _populate_insights_chroma(chroma_client, [seed, i2, i3])

        results = query_corpus(
            seed, "nfl", conn, chroma_client, n_results=5,
            trust_tiers=["core"],
        )
        for r in results:
            assert r.get("trust_tier") == "core"


class TestSummarizeInsight:
    def test_framework_summary(self, conn):
        data = _insert_insight(conn, "i1", fw_name="test_framework")
        # Parse framework_json as the code does before calling _summarize_insight
        data["framework"] = json.loads(data["framework_json"])
        summary = _summarize_insight(data)
        assert summary["insight_id"] == "i1"
        assert summary["framework_name"] == "test_framework"
        assert "mechanism" in summary

    def test_claim_summary(self, conn):
        data = _insert_insight(conn, "i2", insight_type="claim")
        data["claim"] = json.loads(data["claim_json"])
        summary = _summarize_insight(data)
        assert "statement" in summary


class TestBuildCorpusPromptSection:
    def test_includes_seed_and_retrieved(self, conn):
        seed = _insert_insight(conn, "i1")
        r1 = _insert_insight(conn, "i2", analyst="jj")
        section = _build_corpus_prompt_section(seed, [r1])
        assert "SEED INSIGHT" in section
        assert "RETRIEVED CORPUS" in section
        assert "agrees | extends | contradicts | orthogonal" in section

    def test_no_retrieved_notes_thin_evidence(self, conn):
        seed = _insert_insight(conn, "i1")
        section = _build_corpus_prompt_section(seed, [])
        assert "NO RELATED INSIGHTS" in section
        assert "thin evidence" in section.lower()


class TestBuildCorpusTranslatePrompt:
    def test_prompt_contains_all_sections(self, conn):
        seed = _insert_insight(conn, "i1")
        r1 = _insert_insight(conn, "i2", analyst="jj")
        op = OperatorContext(accessible_markets=["polymarket"])

        system, prompt = build_corpus_translate_prompt(
            seed, [r1], '{"confidence":"medium"}', op, "explore",
        )
        assert "CORPUS SYNTHESIS" in system
        assert "supporting_insight_ids" in system
        assert "SEED INSIGHT" in prompt
        assert "polymarket" in prompt


class TestRunTranslateCorpus:
    @patch("research_assistant.stages.translate.call_llm")
    def test_produces_corpus_hypothesis(self, mock_llm, conn, chroma_client, settings):
        mock_llm.return_value = json.dumps([SAMPLE_CORPUS_HYPOTHESIS])

        seed = _insert_insight(conn, "i1", analyst="barrett", fw_name="workload_signal")
        i2 = _insert_insight(conn, "i2", analyst="jj", fw_name="volume_signal")
        _populate_insights_chroma(chroma_client, [seed, i2])

        op = OperatorContext(accessible_markets=["polymarket"])
        hypotheses = run_translate_corpus(
            "i1", "nfl", "explore", op, conn, settings, chroma_client,
        )

        assert len(hypotheses) == 1
        h = hypotheses[0]
        assert h.supporting_insight_ids == ["i2"]
        assert h.contradicting_insight_ids == ["i3"]
        assert h.source_coverage is not None
        assert h.source_coverage.n_sources == 3
        assert "barrett" in h.source_coverage.analysts
        assert h.synthesis_note is not None

    @patch("research_assistant.stages.translate.call_llm")
    def test_single_source_gets_thin_note(self, mock_llm, conn, chroma_client, settings):
        hyp_data = dict(SAMPLE_CORPUS_HYPOTHESIS)
        hyp_data["supporting_insight_ids"] = []
        hyp_data["contradicting_insight_ids"] = []
        hyp_data["source_coverage"] = None
        hyp_data["synthesis_note"] = None
        mock_llm.return_value = json.dumps([hyp_data])

        seed = _insert_insight(conn, "i1", analyst="barrett")
        # No other insights in chroma — empty corpus

        op = OperatorContext()
        hypotheses = run_translate_corpus(
            "i1", "nfl", "explore", op, conn, settings, chroma_client,
        )

        h = hypotheses[0]
        assert h.source_coverage is not None
        assert h.source_coverage.n_sources == 1
        assert "thin evidence" in h.synthesis_note.lower()

    @patch("research_assistant.stages.translate.call_llm")
    def test_save_corpus_fields(self, mock_llm, conn, chroma_client, settings):
        mock_llm.return_value = json.dumps([SAMPLE_CORPUS_HYPOTHESIS])

        seed = _insert_insight(conn, "i1")
        _populate_insights_chroma(chroma_client, [seed])

        op = OperatorContext()
        hypotheses = run_translate_corpus(
            "i1", "nfl", "explore", op, conn, settings, chroma_client,
        )

        ids = save_hypotheses(hypotheses, ["i1"], conn)
        row = get_row(conn, "hypothesis", "hypothesis_id", ids[0])
        assert row is not None
        assert json.loads(row["supporting_insight_ids"]) == ["i2"]
        assert json.loads(row["contradicting_insight_ids"]) == ["i3"]
        coverage = json.loads(row["source_coverage"])
        assert coverage["n_sources"] == 3
        assert row["synthesis_note"] is not None


class TestSelectSeedInsights:
    def test_prefers_unlinked_frameworks(self, conn):
        _insert_insight(conn, "i1", insight_type="framework", fw_name="fw1")
        _insert_insight(conn, "i2", insight_type="framework", fw_name="fw2")
        _insert_insight(conn, "i3", insight_type="claim")

        # Link i1 to a hypothesis
        insert_row(conn, "hypothesis", {
            "hypothesis_id": "h1", "domain_id": "d1",
            "created_at": "2026-01-01T00:00:00Z", "status": "draft",
            "definition_json": "{}", "feasibility_json": "{}",
            "reasoning_chain_json": "{}",
        })
        insert_row(conn, "hypothesis_insight", {
            "hypothesis_id": "h1", "insight_id": "i1",
        })

        seeds = select_seed_insights("nfl", conn)
        assert "i2" in seeds
        assert "i1" not in seeds  # already linked

    def test_falls_back_to_all_active(self, conn):
        _insert_insight(conn, "i1", insight_type="claim")
        # No unlinked frameworks, should fall back
        seeds = select_seed_insights("nfl", conn)
        assert len(seeds) >= 1

    def test_empty_domain(self, conn):
        seeds = select_seed_insights("nonexistent", conn)
        assert seeds == []


class TestHypothesisSchemaCorpusFields:
    def test_valid_with_corpus_fields(self):
        hyp = Hypothesis(domain_id="d1", **SAMPLE_CORPUS_HYPOTHESIS)
        assert hyp.supporting_insight_ids == ["i2"]
        assert hyp.contradicting_insight_ids == ["i3"]
        assert hyp.source_coverage.n_sources == 3
        assert hyp.synthesis_note is not None

    def test_valid_without_corpus_fields(self):
        data = {k: v for k, v in SAMPLE_CORPUS_HYPOTHESIS.items()
                if k not in ("supporting_insight_ids", "contradicting_insight_ids", "source_coverage", "synthesis_note")}
        hyp = Hypothesis(domain_id="d1", **data)
        assert hyp.supporting_insight_ids == []
        assert hyp.source_coverage is None

    def test_source_coverage_model(self):
        sc = SourceCoverage(analysts=["barrett", "jj"], trust_tiers=["core"], n_sources=2)
        assert sc.n_sources == 2
        assert "barrett" in sc.analysts
