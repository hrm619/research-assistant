"""Tests for domain registry loading and contract validation."""

import json
import pytest
from pathlib import Path

from research_assistant.contracts import load_domain_registry, validate_test_definition


@pytest.fixture
def registry_path(tmp_path):
    """Write a valid NFL registry to a temp file."""
    data = {
        "domain": "nfl",
        "contract_version": "1.0.0",
        "metrics_catalog": [
            "yards_per_game_std", "points_per_game_std",
            "third_down_rate_std", "penalty_rate_std",
        ],
        "valid_outcomes": ["ats", "su", "ou"],
        "valid_classification_types": ["quartile", "percentile", "binary", "custom"],
        "valid_lookbacks": ["season_to_date", "last_4"],
        "valid_statistical_tests": ["binomial", "proportion_z", "chi_squared"],
    }
    p = tmp_path / "nfl.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def registry(registry_path):
    return load_domain_registry(registry_path)


def test_load_valid_registry(registry):
    assert registry["domain"] == "nfl"
    assert len(registry["metrics_catalog"]) == 4


def test_load_bad_version(tmp_path):
    data = {"contract_version": "2.0.0", "domain": "nfl",
            "metrics_catalog": [], "valid_outcomes": [],
            "valid_classification_types": [], "valid_lookbacks": [],
            "valid_statistical_tests": []}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported registry"):
        load_domain_registry(p)


def test_load_missing_fields(tmp_path):
    data = {"contract_version": "1.0.0", "domain": "nfl"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing required fields"):
        load_domain_registry(p)


def test_valid_test_definition(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test hypothesis",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "quartile", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert errors == []


def test_unknown_metric(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["fake_metric"],
        "classification": {"type": "quartile", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("Unknown metric" in e for e in errors)


def test_unknown_classification_metric(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "quartile", "metric": "fake_metric"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("Unknown classification metric" in e for e in errors)


def test_invalid_outcome(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "quartile", "metric": "yards_per_game_std"},
        "outcome": "invalid",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("Invalid outcome" in e for e in errors)


def test_invalid_lookback(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "quartile", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "invalid",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("Invalid lookback" in e for e in errors)


def test_percentile_missing_pcts(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "percentile", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("top_pct" in e for e in errors)
    assert any("bottom_pct" in e for e in errors)


def test_binary_missing_threshold(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "binary", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("threshold" in e for e in errors)


def test_custom_missing_boundaries(registry):
    test_def = {
        "hypothesis_name": "test_hyp",
        "description": "A test",
        "version": "1.0.0",
        "metrics": ["yards_per_game_std"],
        "classification": {"type": "custom", "metric": "yards_per_game_std"},
        "outcome": "ats",
        "lookback": "season_to_date",
    }
    errors = validate_test_definition(test_def, registry)
    assert any("boundaries" in e for e in errors)


def test_missing_required_fields(registry):
    test_def = {"version": "1.0.0"}
    errors = validate_test_definition(test_def, registry)
    required_fields = ["hypothesis_name", "description", "metrics",
                       "classification", "outcome", "lookback"]
    for field in required_fields:
        assert any(field in e for e in errors)
