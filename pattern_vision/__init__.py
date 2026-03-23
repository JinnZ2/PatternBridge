"""Pattern vision layer — image analysis via LLM, rubric scoring, and CNN classification."""

from .rubic import PatternRubric, interpret_score
from .prompt_evaluator import PatternPromptEvaluator

__all__ = ["PatternRubric", "interpret_score", "PatternPromptEvaluator"]

# Optional CNN classifier (requires torch)
try:
    from .classifier import PatternClassifier, GARMENT_TYPES, PIECE_NAMES
    __all__ += ["PatternClassifier", "GARMENT_TYPES", "PIECE_NAMES"]
except ImportError:
    pass
