"""
Prompt-based sewing pattern feature evaluator.

Sends pattern piece images to a vision-capable LLM with the pattern
analysis rubric as a structured prompt, then parses the response into
structured PatternFeatures data.

Adapted from hands-lie-detector/prompt/evaluator.py
"""

import base64
import json
import re
from pathlib import Path

from .rubric import PatternRubric, interpret_score


SYSTEM_PROMPT = """You are a sewing pattern analysis system. You analyze images of
sewing pattern pieces to extract structural features needed to reconstruct
the pattern parametrically.

CRITICAL RULES:
- Do NOT guess measurements you cannot see — mark them as null
- Do NOT assume seam allowance unless explicitly stated or brand is identifiable
- Do NOT skip pieces that are partially visible — score what you can see
- DO identify every distinct pattern piece visible in the image
- DO distinguish fold lines from grain lines (they are different features)
- DO note whether the piece is printed/commercial or hand-drawn
- DO flag if multiple sizes are overlaid (graded pattern)

The test: Can you extract enough structural data to reconstruct
this pattern piece at any size from measurements alone?"""


def _build_scoring_prompt(rubric: PatternRubric) -> str:
    """Build the pattern analysis prompt from the rubric."""
    lines = [
        "Analyze this sewing pattern image.",
        "Identify ALL distinct pattern pieces visible.",
        "For each piece, respond with a JSON array where each element has this format:",
        "[",
        "  {",
    ]

    for cat in rubric.categories:
        tier_desc = " | ".join(
            f"{t.min_score}-{t.max_score}: {t.description}"
            for t in cat.tiers
        )
        safe_name = cat.name.lower().replace(" ", "_").replace("&", "and")
        lines.append(
            f'    "{safe_name}": {{"score": <0-{cat.max_points}>, '
            f'"value": <extracted value or null>, "reasoning": "<brief>"}}'
        )
        lines.append(f'    // {cat.question}')
        lines.append(f'    // Tiers: {tier_desc}')

    lines += [
        '    "piece_name": "<name from label or inferred>",',
        '    "piece_number": <integer or null>,',
        '    "cut_quantity": <integer or null>,',
        '    "garment_type": "<pants|dress|skirt|top|jacket|hat|sock|other|unknown>",',
        '    "pattern_brand": "<Butterick|McCall|Vogue|Simplicity|handdrawn|unknown>",',
        '    "seam_allowance_inches": <float or null>,',
        '    "fold_line_present": <true|false>,',
        '    "grain_line_angle_degrees": <float or null>,',
        '    "notch_count": <integer>,',
        '    "dart_count": <integer>,',
        '    "is_graded_multi_size": <true|false>,',
        '    "image_quality_notes": "<anything that limits analysis>"',
        "  }",
        "]",
        "",
        "If only one piece is visible, still return a single-element array.",
        "Extract real values wherever visible. Use null only when genuinely not determinable.",
        "Score based on what is ACTUALLY PRESENT AND READABLE in the image.",
    ]

    return "\n".join(lines)


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Read and base64-encode an image. Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, media_type


def _parse_response(text: str, rubric: PatternRubric) -> list[dict]:
    """
    Extract per-piece feature data from LLM JSON response.
    Returns a list of piece feature dicts.
    """
    # Find JSON array in response
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if not json_match:
        # Try finding a single object and wrap it
        obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if obj_match:
            data = [json.loads(obj_match.group())]
        else:
            raise ValueError(f"Could not find JSON in response: {text[:200]}")
    else:
        data = json.loads(json_match.group())

    if isinstance(data, dict):
        data = [data]

    pieces = []
    for raw in data:
        piece = {}

        # Extract rubric scores
        for cat in rubric.categories:
            safe_name = cat.name.lower().replace(" ", "_").replace("&", "and")
            entry = raw.get(safe_name, {})
            if isinstance(entry, dict):
                piece[f"score_{safe_name}"] = float(entry.get("score", 0))
                piece[f"value_{safe_name}"] = entry.get("value")
                piece[f"reasoning_{safe_name}"] = entry.get("reasoning", "")
            else:
                piece[f"score_{safe_name}"] = float(entry) if entry else 0.0
                piece[f"value_{safe_name}"] = None
                piece[f"reasoning_{safe_name}"] = ""

        # Extract metadata fields
        for field in [
            "piece_name", "piece_number", "cut_quantity", "garment_type",
            "pattern_brand", "seam_allowance_inches", "fold_line_present",
            "grain_line_angle_degrees", "notch_count", "dart_count",
            "is_graded_multi_size", "image_quality_notes",
        ]:
            piece[field] = raw.get(field)

        # Compute total score
        score_keys = [
            f"score_{cat.name.lower().replace(' ', '_').replace('&', 'and')}"
            for cat in rubric.categories
        ]
        piece["total_score"] = sum(piece.get(k, 0) for k in score_keys)
        piece["band_label"] = interpret_score(piece["total_score"])

        pieces.append(piece)

    return pieces


class PatternPromptEvaluator:
    """
    Evaluate sewing pattern images using a vision-capable LLM.

    Supports Anthropic (Claude) and OpenAI (GPT-4o) APIs.
    Returns structured feature data for each pattern piece detected.

    Args:
        provider: "anthropic" or "openai"
        api_key: API key. If None, reads from ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
        model: Model name override.
        rubric: Custom PatternRubric (defaults to v0.1).

    Usage:
        evaluator = PatternPromptEvaluator(provider="anthropic")
        pieces = evaluator.evaluate("patterns/pants_front.jpg")
        for piece in pieces:
            print(piece["piece_name"], piece["band_label"])
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        rubric: PatternRubric | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODELS.get(provider, "")
        self.rubric = rubric or PatternRubric()
        self._scoring_prompt = _build_scoring_prompt(self.rubric)

    def _get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        import os
        env_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key provided. Set {env_var} or pass api_key to constructor."
            )
        return key

    def _call_anthropic(self, image_path: Path) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._get_api_key())
        image_data, media_type = _encode_image(image_path)

        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": self._scoring_prompt,
                    },
                ],
            }],
        )
        return message.content[0].text

    def _call_openai(self, image_path: Path) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self._get_api_key())
        image_data, media_type = _encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {"type": "text", "text": self._scoring_prompt},
                    ],
                },
            ],
        )
        return response.choices[0].message.content

    def evaluate(self, image_path: str | Path) -> list[dict]:
        """
        Analyze a pattern image and extract features for all pieces.

        Args:
            image_path: Path to pattern image (jpg, png, webp).

        Returns:
            List of dicts, one per pattern piece detected.
            Each dict contains rubric scores, extracted values, and metadata.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.provider == "anthropic":
            response_text = self._call_anthropic(image_path)
        elif self.provider == "openai":
            response_text = self._call_openai(image_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return _parse_response(response_text, self.rubric)

    def evaluate_batch(
        self, image_paths: list[str | Path]
    ) -> list[list[dict]]:
        """Evaluate multiple pattern images. Returns list of lists."""
        return [self.evaluate(p) for p in image_paths]

    def evaluate_to_json(
        self, image_path: str | Path, indent: int = 2
    ) -> str:
        """Evaluate and return raw JSON string."""
        return json.dumps(self.evaluate(image_path), indent=indent)
