"""Evaluation script for MathVista predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from evaluation.mathvista import (
    Choice,
    extract_choice_from_response,
    is_correct_choice,
    normalize_free_form,
    summarize_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MathVista predictions.")
    parser.add_argument("--predictions", type=Path, required=True, help="Parquet/JSON file with model outputs.")
    parser.add_argument("--dump-errors", type=Path, help="Optional JSONL file to store incorrect cases.")
    parser.add_argument("--recompute-choice", action="store_true", help="Recompute choice labels from responses.")
    return parser.parse_args()


def _load_predictions(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file {path} not found.")
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_parquet(path)


def _deserialize_choices(value) -> List[Choice]:
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str) and value.strip():
        try:
            entries = json.loads(value)
        except json.JSONDecodeError:
            entries = []
    else:
        entries = []
    choices = []
    for entry in entries:
        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("choice") or entry.get("id") or ""
            text = entry.get("text") or entry.get("value") or ""
        else:
            label = ""
            text = str(entry)
        label = (label or "").strip().upper()
        if not label:
            label = chr(ord("A") + len(choices))
        choices.append(Choice(label=label, text=text))
    return choices


FREE_FORM_PATTERNS = [
    re.compile(r"(?i)(?:final\s+answer|answer|ans)\s*[:=\-]\s*([^\n]+)"),
    re.compile(r"(?i)result\s*[:=\-]\s*([^\n]+)"),
]
BOX_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:/[0-9]+)?")


def extract_free_form_answer(response: str) -> Optional[str]:
    if not response:
        return None
    box = BOX_PATTERN.search(response)
    if box:
        return box.group(1).strip()
    for pattern in FREE_FORM_PATTERNS:
        match = pattern.search(response)
        if match:
            return match.group(1).strip()
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[-1]


def _parse_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = NUMBER_PATTERN.search(text.replace(",", ""))
    if not match:
        return None
    token = match.group(0)
    try:
        if "/" in token:
            num, denom = token.split("/", 1)
            return float(num) / float(denom)
        return float(token)
    except Exception:
        return None


def is_correct_free_form(prediction: Optional[str], answer: str) -> bool:
    if not prediction or not answer:
        return False
    pred_norm = normalize_free_form(prediction)
    ans_norm = normalize_free_form(answer)
    if pred_norm and ans_norm and pred_norm == ans_norm:
        return True
    pred_num = _parse_number(prediction)
    ans_num = _parse_number(answer)
    if pred_num is not None and ans_num is not None:
        if ans_num == 0:
            return abs(pred_num) < 1e-4
        return abs(pred_num - ans_num) / abs(ans_num) < 1e-3
    return False


def main() -> None:
    args = parse_args()
    df = _load_predictions(args.predictions)
    if "answer" not in df.columns:
        raise ValueError("Predictions file must contain an 'answer' column.")

    if "choices" in df.columns:
        df["choices_obj"] = df["choices"].apply(_deserialize_choices)
    else:
        df["choices_obj"] = [[] for _ in range(len(df))]

    has_choices = df["choices_obj"].apply(lambda choices: len(choices) > 0)

    if args.recompute_choice or "prediction" not in df.columns:
        df["choice_prediction"] = df.apply(
            lambda row: extract_choice_from_response(row.get("response", ""), row.get("choices_obj", []))
            if row.get("choices_obj")
            else None,
            axis=1,
        )
    else:
        df["choice_prediction"] = df["prediction"]

    df["free_form_prediction"] = df.apply(
        lambda row: extract_free_form_answer(row.get("response", "")) if not row.get("choices_obj") else None,
        axis=1,
    )

    df["final_prediction"] = df["choice_prediction"]
    df.loc[~has_choices, "final_prediction"] = df.loc[~has_choices, "free_form_prediction"]

    def _row_correct(row):
        if row.get("choices_obj"):
            return is_correct_choice(row.get("final_prediction"), row.get("answer", ""))
        return is_correct_free_form(row.get("final_prediction"), row.get("answer", ""))

    df["is_correct"] = df.apply(_row_correct, axis=1)

    metrics = summarize_accuracy(df.to_dict("records"))
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    if args.dump_errors:
        incorrect = df[~df["is_correct"]]
        incorrect.to_json(args.dump_errors, orient="records", lines=True, force_ascii=False)
        print(f"Wrote {len(incorrect)} incorrect cases to {args.dump_errors}")


if __name__ == "__main__":
    main()
