"""Tests for the pattern_vision layer: rubric and prompt evaluator."""

from __future__ import annotations

import pytest

from pattern_vision.rubic import PatternRubric, Category, Tier, interpret_score


# ── PatternRubric ────────────────────────────────────────────────────────────


class TestPatternRubric:
    def test_init_creates_seven_categories(self):
        rubric = PatternRubric()
        assert len(rubric.categories) == 7

    def test_total_points_is_100(self):
        rubric = PatternRubric()
        assert rubric.total_points == 100

    def test_category_names(self):
        rubric = PatternRubric()
        names = [c.name for c in rubric.categories]
        expected = [
            "Piece Identification",
            "Grain Line",
            "Fold Line",
            "Notch Positions",
            "Dart Definitions",
            "Seam Allowance",
            "Boundary Traceability",
        ]
        assert names == expected

    def test_category_by_name_found(self):
        rubric = PatternRubric()
        cat = rubric.category_by_name("Grain Line")
        assert cat is not None
        assert cat.name == "Grain Line"
        assert cat.max_points == 15

    def test_category_by_name_not_found(self):
        rubric = PatternRubric()
        assert rubric.category_by_name("Nonexistent") is None

    def test_each_category_has_tiers(self):
        rubric = PatternRubric()
        for cat in rubric.categories:
            assert len(cat.tiers) >= 2, f"{cat.name} has fewer than 2 tiers"

    def test_piece_identification_is_20_points(self):
        rubric = PatternRubric()
        cat = rubric.category_by_name("Piece Identification")
        assert cat.max_points == 20


# ── interpret_score ──────────────────────────────────────────────────────────


class TestInterpretScore:
    def test_perfect_score(self):
        result = interpret_score(100)
        assert "Complete" in result

    def test_zero_score(self):
        result = interpret_score(0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mid_range(self):
        result = interpret_score(50)
        assert isinstance(result, str)

    def test_good_score(self):
        result = interpret_score(75)
        assert "Good" in result or "good" in result.lower()

    def test_returns_string_for_all_ranges(self):
        for score in [0, 10, 25, 40, 55, 70, 85, 100]:
            result = interpret_score(score)
            assert isinstance(result, str)
            assert len(result) > 0


# ── PatternPromptEvaluator (no API calls) ────────────────────────────────────


class TestPromptEvaluator:
    """Test evaluator construction and helper functions (no LLM calls)."""

    def test_import(self):
        from pattern_vision.prompt_evaluator import PatternPromptEvaluator
        assert PatternPromptEvaluator is not None

    def test_init_default_provider(self):
        from pattern_vision.prompt_evaluator import PatternPromptEvaluator
        evaluator = PatternPromptEvaluator(provider="anthropic", api_key="test-key")
        assert evaluator.provider == "anthropic"
        assert evaluator.rubric is not None

    def test_init_openai_provider(self):
        from pattern_vision.prompt_evaluator import PatternPromptEvaluator
        evaluator = PatternPromptEvaluator(provider="openai", api_key="test-key")
        assert evaluator.provider == "openai"

    def test_build_scoring_prompt(self):
        from pattern_vision.prompt_evaluator import _build_scoring_prompt
        rubric = PatternRubric()
        prompt = _build_scoring_prompt(rubric)
        # Category names are lowercased and underscored in the prompt
        assert "piece_identification" in prompt or "piece" in prompt.lower()
        assert "grain_line" in prompt or "grain" in prompt.lower()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_parse_response_valid_json(self):
        from pattern_vision.prompt_evaluator import _parse_response
        rubric = PatternRubric()
        response_text = '''```json
[
  {
    "piece_name": "FRONT",
    "piece_number": 1,
    "piece_identification": {"score": 15, "value": "FRONT", "reasoning": "clear label"},
    "grain_line": {"score": 10, "value": null, "reasoning": "visible"},
    "fold_line": {"score": 12, "value": null, "reasoning": "none"},
    "notch_positions": {"score": 10, "value": null, "reasoning": "2 visible"},
    "dart_definitions": {"score": 8, "value": null, "reasoning": "none"},
    "seam_allowance": {"score": 7, "value": 0.625, "reasoning": "standard"},
    "boundary_traceability": {"score": 8, "value": null, "reasoning": "clear"}
  }
]
```'''
        result = _parse_response(response_text, rubric)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["piece_name"] == "FRONT"
