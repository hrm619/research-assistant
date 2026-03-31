"""Domain registry loading and contract validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_domain_registry(path: Path) -> dict:
    """Load and validate a domain registry JSON file."""
    with open(path) as f:
        registry = json.load(f)

    version = registry.get("contract_version", "")
    if not version.startswith("1."):
        raise ValueError(
            f"Unsupported registry contract_version: {version!r}. "
            "Expected major version 1."
        )

    required = ["domain", "metrics_catalog", "valid_outcomes",
                 "valid_classification_types", "valid_lookbacks",
                 "valid_statistical_tests"]
    missing = [k for k in required if k not in registry]
    if missing:
        raise ValueError(f"Registry missing required fields: {missing}")

    return registry


def validate_test_definition(test_def: dict, registry: dict) -> list[str]:
    """Validate a test_definition dict against a domain registry.

    Returns a list of validation error strings. Empty list means valid.
    """
    errors: list[str] = []
    catalog = set(registry["metrics_catalog"])

    # Check metrics list
    for metric in test_def.get("metrics", []):
        if metric not in catalog:
            errors.append(f"Unknown metric: {metric!r}")

    # Check classification
    cls = test_def.get("classification", {})
    cls_metric = cls.get("metric", "")
    if cls_metric and cls_metric not in catalog:
        errors.append(f"Unknown classification metric: {cls_metric!r}")

    cls_type = cls.get("type", "")
    if cls_type and cls_type not in registry["valid_classification_types"]:
        errors.append(f"Invalid classification type: {cls_type!r}")

    # Conditional requirements
    if cls_type == "percentile":
        if cls.get("top_pct") is None:
            errors.append("percentile classification requires top_pct")
        if cls.get("bottom_pct") is None:
            errors.append("percentile classification requires bottom_pct")
    elif cls_type == "binary":
        if cls.get("threshold") is None:
            errors.append("binary classification requires threshold")
    elif cls_type == "custom":
        if not cls.get("boundaries"):
            errors.append("custom classification requires boundaries")

    # Check outcome
    outcome = test_def.get("outcome", "")
    if outcome and outcome not in registry["valid_outcomes"]:
        errors.append(f"Invalid outcome: {outcome!r}")

    # Check lookback
    lookback = test_def.get("lookback", "")
    if lookback and lookback not in registry["valid_lookbacks"]:
        errors.append(f"Invalid lookback: {lookback!r}")

    # Check statistical_test
    stat_test = test_def.get("statistical_test", "binomial")
    if stat_test not in registry["valid_statistical_tests"]:
        errors.append(f"Invalid statistical_test: {stat_test!r}")

    # Check required fields
    required = ["hypothesis_name", "description", "metrics", "classification",
                 "outcome", "lookback"]
    for field in required:
        if not test_def.get(field):
            errors.append(f"Missing required field: {field!r}")

    return errors
