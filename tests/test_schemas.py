import pytest
from pydantic import ValidationError

from research_assistant.schemas import (
    Claim,
    ContentItem,
    DomainBriefContent,
    Feasibility,
    Framework,
    Hypothesis,
    HypothesisDefinition,
    Insight,
    ReasoningChain,
)


def _framework(**overrides):
    defaults = {
        "name": "test_framework",
        "author_attribution": "Author",
        "mechanism": "When X happens, Y follows because Z",
        "conditions": ["X is true"],
        "predictions": ["Y will increase"],
        "assumptions": ["Z holds"],
        "evidence_cited": ["paper1"],
        "confidence_language": "likely",
    }
    defaults.update(overrides)
    return Framework(**defaults)


def _claim(**overrides):
    defaults = {
        "statement": "Fed will hold rates",
        "reasoning": "Inflation data supports this",
        "timeframe": "Q3 2026",
        "falsifiable": True,
        "falsification_trigger": "CPI exceeds 4%",
    }
    defaults.update(overrides)
    return Claim(**defaults)


class TestDomainBriefContent:
    def test_valid(self):
        brief = DomainBriefContent(
            market_mechanics={
                "instrument_type": "binary",
                "settlement": "cash",
                "liquidity_profile": "thin",
                "fee_structure": "2%",
                "position_types": ["long", "short"],
                "known_biases": ["retail skew"],
            },
            game_theory={
                "participant_types": ["retail"],
                "information_asymmetries": ["insider data"],
                "common_mistakes": ["overconfidence"],
            },
            current_meta={
                "dominant_narratives": ["soft landing"],
                "consensus_view": "rates will hold",
                "contrarian_angles": ["recession risk"],
            },
            analogies=[
                {"known_domain": "sports", "mapping": "similar odds", "where_analogy_breaks": "liquidity"},
            ],
            key_data_sources=["FRED"],
            open_questions=["What drives volume?"],
            confidence="medium",
        )
        assert brief.confidence == "medium"

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            DomainBriefContent(
                market_mechanics={
                    "instrument_type": "binary", "settlement": "cash",
                    "liquidity_profile": "thin", "fee_structure": "2%",
                    "position_types": ["long"], "known_biases": [],
                },
                game_theory={"participant_types": [], "information_asymmetries": [], "common_mistakes": []},
                current_meta={"dominant_narratives": [], "consensus_view": "", "contrarian_angles": []},
                analogies=[],
                key_data_sources=[],
                open_questions=[],
                confidence="very high",
            )


class TestInsight:
    def test_framework_insight_valid(self):
        insight = Insight(
            content_id="c1",
            source_id="s1",
            domain_id="d1",
            insight_type="framework",
            framework=_framework(),
            source_quote_ref="paragraph 3",
        )
        assert insight.insight_type == "framework"

    def test_framework_insight_missing_framework(self):
        with pytest.raises(ValidationError, match="Framework insight must include framework data"):
            Insight(
                content_id="c1",
                source_id="s1",
                domain_id="d1",
                insight_type="framework",
                source_quote_ref="paragraph 3",
            )

    def test_framework_empty_mechanism(self):
        with pytest.raises(ValidationError, match="mechanism must not be empty"):
            Insight(
                content_id="c1",
                source_id="s1",
                domain_id="d1",
                insight_type="framework",
                framework=_framework(mechanism=""),
                source_quote_ref="paragraph 3",
            )

    def test_framework_empty_conditions(self):
        with pytest.raises(ValidationError, match="conditions must not be empty"):
            Insight(
                content_id="c1",
                source_id="s1",
                domain_id="d1",
                insight_type="framework",
                framework=_framework(conditions=[]),
                source_quote_ref="paragraph 3",
            )

    def test_claim_insight_valid(self):
        insight = Insight(
            content_id="c1",
            source_id="s1",
            domain_id="d1",
            insight_type="claim",
            claim=_claim(),
            source_quote_ref="paragraph 5",
        )
        assert insight.claim.falsifiable is True

    def test_claim_insight_missing_claim(self):
        with pytest.raises(ValidationError, match="Claim insight must include claim data"):
            Insight(
                content_id="c1",
                source_id="s1",
                domain_id="d1",
                insight_type="claim",
                source_quote_ref="paragraph 5",
            )

    def test_claim_falsifiable_without_trigger_becomes_false(self):
        claim = _claim(falsifiable=True, falsification_trigger=None)
        assert claim.falsifiable is False


class TestHypothesis:
    def _hypothesis(self, **overrides):
        defaults = {
            "domain_id": "d1",
            "definition": {
                "name": "test",
                "statement": "X predicts Y",
                "factor": "X metric",
                "classification": "quartile",
                "outcome_measure": "return",
                "timeframe": "30d",
                "data_required": ["price data"],
                "data_available": True,
                "market_expression": "Kalshi binary on X",
            },
            "feasibility": {
                "data_gap": [],
                "knowledge_gap": [],
                "estimated_testability": "high",
            },
            "reasoning_chain": {
                "from_insight": "Expert says X causes Y",
                "translation_logic": "If X then Y, measurable via Z",
                "assumptions_added": ["Z is a good proxy"],
                "weaknesses": ["Small sample size"],
            },
        }
        defaults.update(overrides)
        return Hypothesis(**defaults)

    def test_valid(self):
        h = self._hypothesis()
        assert h.status == "draft"

    def test_empty_weaknesses(self):
        with pytest.raises(ValidationError, match="weaknesses must not be empty"):
            self._hypothesis(reasoning_chain={
                "from_insight": "x",
                "translation_logic": "y",
                "assumptions_added": ["a"],
                "weaknesses": [],
            })

    def test_empty_assumptions_added(self):
        with pytest.raises(ValidationError, match="assumptions_added must not be empty"):
            self._hypothesis(reasoning_chain={
                "from_insight": "x",
                "translation_logic": "y",
                "assumptions_added": [],
                "weaknesses": ["w"],
            })


class TestContentItem:
    def test_valid(self):
        item = ContentItem(
            source_id="s1",
            content_type="transcript",
            title="Test Video",
            author="Author",
            raw_text="Some text content here",
            word_count=4,
        )
        assert item.processing_status == "success"
        assert item.format_metadata.has_sections is False

    def test_json_round_trip(self):
        item = ContentItem(
            source_id="s1",
            content_type="transcript",
            title="Test",
            author="Author",
            raw_text="text",
            word_count=1,
        )
        data = item.model_dump()
        item2 = ContentItem(**data)
        assert item2.content_id == item.content_id
