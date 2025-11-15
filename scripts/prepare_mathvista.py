#!/usr/bin/env python3
"""Converts MathVista annotations into the parquet format expected by evaluation scripts."""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MathVista split for VERL evaluation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-json", type=Path, help="Local JSON/JSONL/parquet file with MathVista annotations.")
    source.add_argument("--hf-dataset", help="Optional Hugging Face dataset id, e.g. AI4Math/MathVista.")

    parser.add_argument("--split", default="testmini", help="Dataset split to use (when loading from HF).")
    parser.add_argument("--output", type=Path, required=True, help="Destination parquet path.")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--choices-key", default="choices")
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--decoded-image-key", default="decoded_image")
    parser.add_argument("--id-key", default="question_id")
    parser.add_argument("--data-source", default=None, help="Optional override for the data_source column.")
    parser.add_argument("--image-root", type=Path, help="Root dir for images; used to compute relative paths.")
    parser.add_argument("--export-images-to", type=Path, help="Optional directory to export decoded images.")
    parser.add_argument("--limit", type=int, help="Optional record limit for debugging.")
    parser.add_argument("--metadata-keys", nargs="*", default=["category", "subcategory", "type"])
    return parser.parse_args()


def _read_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.hf_dataset:
        if load_dataset is None:
            raise RuntimeError("datasets package not installed; install `datasets` or use --source-json.")
        dataset = load_dataset(args.hf_dataset, split=args.split)
        return dataset.to_list()

    source_path = Path(args.source_json)
    if source_path.suffix == ".jsonl":
        df = pd.read_json(source_path, lines=True)
        return df.to_dict("records")
    if source_path.suffix == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
        return df.to_dict("records")
    raise ValueError(f"Unsupported source file format: {source_path.suffix}")


def _normalize_image_paths(raw_value: Any, image_root: Optional[Path]) -> List[str]:
    def extract_path(entry: Any) -> Optional[str]:
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            return entry.get("path") or entry.get("file_name") or entry.get("image") or entry.get("value")
        return str(entry)

    rel_paths: List[str] = []
    raw_list: List[Any]
    if raw_value is None:
        raw_list = []
    elif isinstance(raw_value, list):
        raw_list = raw_value
    else:
        raw_list = [raw_value]

    for item in raw_list:
        resolved = extract_path(item)
        if not resolved:
            continue
        path = Path(resolved)
        if image_root and path.is_absolute():
            try:
                rel_paths.append(str(path.relative_to(image_root)))
            except ValueError:
                rel_paths.append(str(path))
        else:
            rel_paths.append(str(path))
    return rel_paths


def _extract_image_objects(raw_value: Any) -> List[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return raw_value
    return [raw_value]


def _save_image_object(obj: Any, target_path: Path) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(obj, str):
            src = Path(obj)
            if src.exists():
                shutil.copy(src, target_path)
                return True
            return False
        if isinstance(obj, dict):
            if obj.get("path"):
                src = Path(obj["path"])
                if src.exists():
                    shutil.copy(src, target_path)
                    return True
            if obj.get("bytes"):
                target_path.write_bytes(obj["bytes"])
                return True
            return False
        if hasattr(obj, "save"):
            obj.save(target_path)
            return True
        if hasattr(obj, "to_pil"):
            obj.to_pil().save(target_path)
            return True
    except Exception:
        return False
    return False


def main() -> None:
    args = parse_args()
    raw_records = _read_records(args)
    if args.limit:
        raw_records = raw_records[: args.limit]

    records: List[Dict[str, Any]] = []
    for record in raw_records:
        question = record.get(args.question_key)
        answer = record.get(args.answer_key)
        if question is None or answer is None:
            continue

        question_id = record.get(args.id_key) or record.get("question_id") or record.get("qid") or record.get("pid")
        if not question_id:
            question_id = f"mathvista-{len(records)}"

        image_rel_paths = _normalize_image_paths(record.get(args.image_key), args.image_root)
        decoded_images = _extract_image_objects(record.get(args.decoded_image_key))
        if args.export_images_to and decoded_images:
            export_root = args.export_images_to
            export_root.mkdir(parents=True, exist_ok=True)
            exported_paths: List[str] = []
            for idx, (rel_path, image_obj) in enumerate(zip_longest(image_rel_paths, decoded_images)):
                if image_obj is None:
                    continue
                rel_path = rel_path or f"{question_id}_{idx}.png"
                target_path = export_root / rel_path
                if _save_image_object(image_obj, target_path):
                    exported_paths.append(rel_path)
            if exported_paths:
                image_rel_paths = exported_paths
        metadata = {key: record.get(key) for key in args.metadata_keys if key in record and record.get(key) is not None}
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        records.append(
            {
                "question_id": str(question_id),
                "question": str(question),
                "answer": str(answer),
                "choices": record.get(args.choices_key),
                "image_paths": image_rel_paths,
                "data_source": args.data_source or f"MathVista/{args.split}",
                "metadata": metadata_json,
            }
        )

    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
