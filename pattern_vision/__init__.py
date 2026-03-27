"""Pattern vision layer — image analysis via LLM, rubric scoring, and CNN classification."""

from .rubic import PatternRubric, interpret_score
from .prompt_evaluator import PatternPromptEvaluator

__all__ = ["PatternRubric", "interpret_score", "PatternPromptEvaluator"]

# Optional CNN classifier + dataset + training (requires torch)
try:
    from .classifier import PatternClassifier, GARMENT_TYPES, PIECE_NAMES
    from .dataset import PatternDataset, load_annotations
    from .train import train as train_classifier, MultiTaskLoss
    __all__ += [
        "PatternClassifier", "GARMENT_TYPES", "PIECE_NAMES",
        "PatternDataset", "load_annotations",
        "train_classifier", "MultiTaskLoss",
    ]
except ImportError:
    pass
