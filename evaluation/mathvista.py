"""Helpers for preparing MathVista prompts and evaluating predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

CHOICE_INLINE_PATTERN = re.compile(r"^\s*([A-Z])[\.\)]\s*(.+)$")
LETTER_PATTERN = re.compile(r"\b([A-Z])\b")


@dataclass
class Choice:
    """Represents a single multiple-choice option."""

    label: str
    text: str

    def as_prompt_fragment(self) -> str:
        text = self.text.strip()
        return f"{self.label}. {text}" if text else self.label


@dataclass
class MathVistaSample:
    """Structured entry expected by the generation script."""

    question_id: str
    question: str
    answer: str
    data_source: str
    image_paths: List[str]
    choices: List[Choice]
    metadata: Dict[str, Any]


def _as_list(value: Any) -> List[Any]:
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [value]
    return [value]


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _parse_choices(raw_choices: Any) -> List[Choice]:
    parsed_choices: List[Choice] = []
    for idx, entry in enumerate(_as_list(raw_choices)):
        label = None
        text = ""
        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("choice") or entry.get("id")
            text = entry.get("text") or entry.get("value") or entry.get("desc") or ""
        elif isinstance(entry, str):
            match = CHOICE_INLINE_PATTERN.match(entry)
            if match:
                label = match.group(1)
                text = match.group(2)
            else:
                text = entry
        if not label:
            label = chr(ord("A") + idx)
        parsed_choices.append(Choice(label=label.strip().upper(), text=text))
    return parsed_choices


def load_mathvista_samples(table_path: Path, limit: Optional[int] = None) -> List[MathVistaSample]:
    """Loads MathVista samples from a parquet/JSON file."""

    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Dataset file {table_path} not found.")

    if table_path.suffix == ".json":
        df = pd.read_json(table_path)
    elif table_path.suffix == ".jsonl":
        df = pd.read_json(table_path, lines=True)
    else:
        df = pd.read_parquet(table_path)

    missing = [col for col in ("question_id", "question", "answer") if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset file {table_path} misses required columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            MathVistaSample(
                question_id=str(row["question_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]).strip(),
                data_source=str(row.get("data_source", "MathVista")),
                image_paths=[str(path) for path in _as_list(row.get("image_paths"))],
                choices=_parse_choices(row.get("choices")),
                metadata=_as_dict(row.get("metadata")),
            )
        )

    if limit is not None:
        rows = rows[:limit]
    return rows


def format_question(sample: MathVistaSample, include_rationale_hint: bool = True) -> str:
    """Formats the textual part of the MathVista question that will be fed to Qwen."""

    prompt_lines = [sample.question.strip()]
    if sample.choices:
        prompt_lines.append("")
        prompt_lines.append("Choices:")
        for choice in sample.choices:
            prompt_lines.append(f"- {choice.as_prompt_fragment()}")
        prompt_lines.append("")
        prompt_lines.append("Answer with the letter of the correct option and a one sentence justification.")

    elif include_rationale_hint:
        prompt_lines.append("")
        prompt_lines.append("Provide your final numeric/text answer after reasoning about the visual content.")

    return "\n".join(prompt_lines).strip()


def normalize_free_form(text: str) -> str:
    return re.sub(r"[\s\t\n]+", " ", text.strip().lower())


def extract_choice_from_response(response: str, valid_choices: Sequence[Choice]) -> Optional[str]:
    """Extracts a choice label from the model response."""

    if not response or not valid_choices:
        return None

    response_upper = response.strip().upper()
    # Look for explicit "Answer: X" patterns
    answer_idx = response_upper.find("ANSWER")
    if answer_idx != -1:
        tail = response_upper[answer_idx:]
        match = LETTER_PATTERN.search(tail)
        if match and match.group(1) in {c.label for c in valid_choices}:
            return match.group(1)

    # Fallback: first standalone capital letter
    for match in LETTER_PATTERN.finditer(response_upper):
        letter = match.group(1)
        if letter in {c.label for c in valid_choices}:
            return letter

    # If the model copies the entire option text, align if possible
    normalized = normalize_free_form(response)
    for choice in valid_choices:
        if normalize_free_form(choice.text) in normalized:
            return choice.label

    return None


def is_correct_choice(prediction: Optional[str], answer: str) -> bool:
    if prediction is None:
        return False
    if not answer:
        return False
    return prediction.strip().upper()[0] == answer.strip().upper()[0]


def summarize_accuracy(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates accuracy per data_source and overall."""

    totals: Dict[str, Dict[str, int]] = {}
    for row in rows:
        source = row.get("data_source", "MathVista")
        totals.setdefault(source, {"correct": 0, "total": 0})
        totals[source]["total"] += 1
        totals[source]["correct"] += int(bool(row.get("is_correct", False)))

    metrics = {}
    overall_correct = sum(stat["correct"] for stat in totals.values())
    overall_total = sum(stat["total"] for stat in totals.values()) or 1
    metrics["overall_accuracy"] = overall_correct / overall_total
    for source, stat in totals.items():
        metrics[f"accuracy/{source}"] = stat["correct"] / max(stat["total"], 1)
    return metrics
