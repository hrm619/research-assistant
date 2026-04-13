from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _uuid() -> str:
    return str(uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Orient / DomainBrief ---


class MarketMechanics(BaseModel):
    instrument_type: str
    settlement: str
    liquidity_profile: str
    fee_structure: str
    position_types: list[str]
    known_biases: list[str]


class GameTheory(BaseModel):
    participant_types: list[str]
    information_asymmetries: list[str]
    common_mistakes: list[str]


class CurrentMeta(BaseModel):
    dominant_narratives: list[str]
    consensus_view: str
    contrarian_angles: list[str]


class Analogy(BaseModel):
    known_domain: str
    mapping: str
    where_analogy_breaks: str


class DomainBriefContent(BaseModel):
    market_mechanics: MarketMechanics
    game_theory: GameTheory
    current_meta: CurrentMeta
    analogies: list[Analogy]
    key_data_sources: list[str]
    open_questions: list[str]
    confidence: Literal["low", "medium", "high"]


class OrientInput(BaseModel):
    domain_name: str
    market_type: str
    operator_known_domains: list[str]
    seed_questions: list[str]
    seed_sources: list[str] = Field(default_factory=list)


# --- Insight ---


class Framework(BaseModel):
    name: str
    author_attribution: str
    mechanism: str
    conditions: list[str]
    predictions: list[str]
    assumptions: list[str]
    evidence_cited: list[str]
    confidence_language: str


class Claim(BaseModel):
    statement: str
    reasoning: str
    timeframe: str | None = None
    falsifiable: bool
    falsification_trigger: str | None = None

    @model_validator(mode="after")
    def check_falsification(self):
        if not self.falsifiable and self.falsification_trigger:
            raise ValueError("Non-falsifiable claim should not have a falsification_trigger")
        if self.falsifiable and not self.falsification_trigger:
            self.falsifiable = False
        return self


class Insight(BaseModel):
    insight_id: str = Field(default_factory=_uuid)
    content_id: str
    content_item_ref: str = ""
    source_id: str = ""
    domain_id: str
    extracted_at: str = Field(default_factory=_now_iso)
    insight_type: Literal["framework", "claim", "observation"]
    framework: Framework | None = None
    claim: Claim | None = None
    source_quote_ref: str
    operator_note: str | None = None
    status: Literal["active", "merged", "discarded"] = "active"
    analyst: str = ""
    trust_tier: str = ""
    content_source: Literal["ra", "kb"] = "ra"

    @model_validator(mode="after")
    def check_type_data(self):
        if self.insight_type == "framework" and self.framework is None:
            raise ValueError("Framework insight must include framework data")
        if self.insight_type == "claim" and self.claim is None:
            raise ValueError("Claim insight must include claim data")
        if self.insight_type == "framework" and self.framework and not self.framework.mechanism:
            raise ValueError("Framework mechanism must not be empty")
        if self.insight_type == "framework" and self.framework and not self.framework.conditions:
            raise ValueError("Framework conditions must not be empty")
        return self


# --- TestDefinition (machine-readable for factor-research) ---


class TestDefinition(BaseModel):
    hypothesis_name: str
    description: str
    version: str = "1.0.0"
    metrics: list[str]
    classification: dict
    outcome: str
    lookback: str
    filters: dict | None = None
    min_sample_size: int = 50
    statistical_test: str = "binomial"
    significance_threshold: float = 0.05
    comparison_buckets: list[str] = Field(default_factory=list)
    time_windows: list[dict] = Field(default_factory=list)


# --- Hypothesis ---


class HypothesisDefinition(BaseModel):
    name: str
    statement: str
    factor: str
    classification: str
    outcome_measure: str
    timeframe: str
    data_required: list[str]
    data_available: bool
    market_expression: str


class Feasibility(BaseModel):
    data_gap: list[str] = Field(default_factory=list)
    knowledge_gap: list[str] = Field(default_factory=list)
    minimum_sample_size: int | None = None
    estimated_testability: Literal["high", "medium", "low"]


class ReasoningChain(BaseModel):
    from_insight: str
    translation_logic: str
    assumptions_added: list[str]
    weaknesses: list[str]

    @model_validator(mode="after")
    def check_non_empty(self):
        if not self.weaknesses:
            raise ValueError("weaknesses must not be empty")
        if not self.assumptions_added:
            raise ValueError("assumptions_added must not be empty")
        return self


class SourceCoverage(BaseModel):
    analysts: list[str] = Field(default_factory=list)
    trust_tiers: list[str] = Field(default_factory=list)
    n_sources: int = 0


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=_uuid)
    domain_id: str
    created_at: str = Field(default_factory=_now_iso)
    status: Literal["draft", "review", "accepted", "rejected", "tested"] = "draft"
    definition: HypothesisDefinition
    feasibility: Feasibility
    reasoning_chain: ReasoningChain
    test_definition: TestDefinition | None = None
    operator_note: str | None = None
    supporting_insight_ids: list[str] = Field(default_factory=list)
    contradicting_insight_ids: list[str] = Field(default_factory=list)
    source_coverage: SourceCoverage | None = None
    synthesis_note: str | None = None


# --- Stage Inputs ---


class DistillInput(BaseModel):
    content_id: str
    domain_id: str
    extraction_mode: Literal["framework", "claim", "both"] = "both"
    operator_focus: str | None = None


class OperatorContext(BaseModel):
    accessible_markets: list[str] = Field(default_factory=list)
    available_data_sources: list[str] = Field(default_factory=list)
    capital_constraints: str | None = None
    review_cadence: str = "weekly"
    known_domains: list[str] = Field(default_factory=list)


class TranslateInput(BaseModel):
    insight_ids: list[str]
    domain_id: str
    operator_context: OperatorContext = Field(default_factory=OperatorContext)
    translation_mode: Literal["explore", "commit"] = "explore"
