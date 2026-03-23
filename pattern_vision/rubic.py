"""
PatternRubric: Scoring rubric for sewing pattern feature detection.

Seven categories totaling 100 points, designed to mirror the
ScoringRubric structure from hands-lie-detector for compatibility.

Categories detect the structural features needed to reconstruct
a pattern piece parametrically from an image.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Tier:
    min_score: int
    max_score: int
    description: str


@dataclass
class Category:
    name: str
    max_points: int
    question: str
    tiers: List[Tier]


@dataclass
class PatternRubric:
    """
    100-point rubric for extracting structural features from
    sewing pattern images. Designed for use with PromptEvaluator
    and PatternClassifier.
    """

    categories: List[Category] = field(default_factory=list)

    def __post_init__(self):
        self.categories = [

            Category(
                name="Piece Identification",
                max_points=20,
                question="Can you identify the piece name, number, cut quantity, and garment type?",
                tiers=[
                    Tier(0, 4,   "No readable label or piece number"),
                    Tier(5, 9,   "Partial — name or number visible but not both"),
                    Tier(10, 14, "Name and number clear, cut quantity ambiguous"),
                    Tier(15, 17, "Name, number, cut quantity all clear"),
                    Tier(18, 20, "Full ID: name, number, quantity, garment type, and size range"),
                ]
            ),

            Category(
                name="Grain Line",
                max_points=15,
                question="Is a grain line present, and can its direction and angle be determined?",
                tiers=[
                    Tier(0, 2,   "No grain line visible"),
                    Tier(3, 6,   "Line visible but direction arrows missing or ambiguous"),
                    Tier(7, 10,  "Grain line clear with direction, angle not precisely measurable"),
                    Tier(11, 13, "Grain line with arrows, approximate angle determinable"),
                    Tier(14, 15, "Grain line fully measurable: position, direction, angle from boundary"),
                ]
            ),

            Category(
                name="Fold Line",
                max_points=15,
                question="Is there a fold line, and can its axis and position be precisely located?",
                tiers=[
                    Tier(0, 2,   "No fold line — piece is cut open (not on fold)"),
                    Tier(3, 6,   "Fold line label present but position unclear"),
                    Tier(7, 10,  "Fold line visible, axis determinable, position approximate"),
                    Tier(11, 13, "Fold line clearly positioned along one boundary edge"),
                    Tier(14, 15, "Fold line fully mapped: axis, position, confirmed by boundary geometry"),
                ]
            ),

            Category(
                name="Notch Positions",
                max_points=15,
                question="How many notches are present and can their positions on the boundary be located?",
                tiers=[
                    Tier(0, 2,   "No notches visible or identifiable"),
                    Tier(3, 6,   "Some notches visible but positions not mappable"),
                    Tier(7, 10,  "Notch count accurate, approximate boundary positions determinable"),
                    Tier(11, 13, "All notches located with good boundary position accuracy"),
                    Tier(14, 15, "All notches precisely located with seam alignment context"),
                ]
            ),

            Category(
                name="Dart Definitions",
                max_points=15,
                question="Are darts present, and can apex, leg lines, and depth be determined?",
                tiers=[
                    Tier(0, 2,   "No darts present or none visible"),
                    Tier(3, 6,   "Dart marks visible but shape not determinable"),
                    Tier(7, 10,  "Dart location clear, apex approximate, legs not fully traced"),
                    Tier(11, 13, "Dart apex and legs traceable, depth approximate"),
                    Tier(14, 15, "Full dart definition: apex, both leg lines, depth, intake measurement"),
                ]
            ),

            Category(
                name="Seam Allowance",
                max_points=10,
                question="Is the seam allowance specified, and is it consistent or variable across edges?",
                tiers=[
                    Tier(0, 1,  "No seam allowance information"),
                    Tier(2, 4,  "Standard assumption only (e.g. 5/8\" implied by pattern brand)"),
                    Tier(5, 6,  "Seam allowance stated globally but edge exceptions unclear"),
                    Tier(7, 8,  "Global seam allowance stated with some edge-specific notes"),
                    Tier(9, 10, "Full seam allowance map: global value + all edge-specific exceptions"),
                ]
            ),

            Category(
                name="Boundary Traceability",
                max_points=10,
                question="How completely and accurately can the full outer boundary of the piece be traced?",
                tiers=[
                    Tier(0, 1,  "Boundary not traceable — obscured, cut off, or overlapping"),
                    Tier(2, 4,  "Partial boundary only — significant sections missing"),
                    Tier(5, 6,  "Most of boundary traceable with some ambiguous sections"),
                    Tier(7, 8,  "Full boundary traceable with minor curve ambiguity"),
                    Tier(9, 10, "Complete clean boundary: all edges, curves, corners fully defined"),
                ]
            ),

        ]

    def category_by_name(self, name: str) -> Optional[Category]:
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None

    @property
    def total_points(self) -> int:
        return sum(c.max_points for c in self.categories)


# Interpretation bands — maps total score to assessment label
PATTERN_BANDS = [
    (0,  30,  "Unusable — insufficient data to reconstruct piece"),
    (31, 50,  "Partial — major features missing, reconstruction unreliable"),
    (51, 65,  "Basic — primary shape recoverable, fine details uncertain"),
    (66, 80,  "Good — piece reconstructable with minor assumptions"),
    (81, 90,  "Strong — full reconstruction possible with high confidence"),
    (91, 100, "Complete — all features precisely defined, ready for encoding"),
]


def interpret_score(total: float) -> str:
    for low, high, label in PATTERN_BANDS:
        if low <= total <= high:
            return label
    return "Unknown"
