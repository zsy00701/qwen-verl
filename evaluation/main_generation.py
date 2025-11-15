"""Runs MathVista generation with Qwen2.5-VL using VERL-inspired scaffolding."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from verl.utils import hf_processor
from verl.utils.dataset.vision_utils import process_image

from evaluation.mathvista import (
    Choice,
    extract_choice_from_response,
    format_question,
    load_mathvista_samples,
)
from tools.vision_toolbox import ToolView, VisionToolbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MathVista generation with Qwen2.5-VL.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared MathVista parquet/json file.")
    parser.add_argument("--image-root", type=Path, required=True, help="Directory that contains MathVista images.")
    parser.add_argument("--output", type=Path, required=True, help="Destination parquet file for generations.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id or local path.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Precision for the vision-language model.",
    )
    parser.add_argument("--device-map", default="auto", help="device_map argument for from_pretrained.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, help="Optional number of samples to run (debug).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default="You are a precise math tutor. Reason about the image before answering.")
    parser.add_argument("--use-tools", action="store_true", help="Enable the heuristic vision toolbox.")
    parser.add_argument("--tool-max-views", type=int, default=2, help="Max number of additional tool-generated views.")
    parser.add_argument(
        "--trust-remote-code",
        dest="trust_remote_code",
        action="store_true",
        help="Enable custom modeling code provided by the model repo (recommended for Qwen VL).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable remote code if you are using a fully upstream-compatible checkpoint.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--save-every", type=int, default=200, help="Periodic flush interval (rows).")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _infer_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for device in device_map.values():
            if device == "disk":
                continue
            return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_messages(
    sample,
    system_prompt: str,
    image_root: Path,
    toolbox: Optional[VisionToolbox],
) -> tuple[List[Dict], List[Dict], List[Image.Image]]:
    def to_absolute(path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = image_root / path
        return path

    user_content: List[Dict] = []
    tool_log: List[Dict] = []
    image_inputs: List[Image.Image] = []

    for image_path in sample.image_paths:
        abs_path = to_absolute(image_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Image {abs_path} not found for question {sample.question_id}")
        with Image.open(abs_path) as pil_img:
            base_image = process_image(pil_img)

        if toolbox:
            views = toolbox.apply(base_image, sample.question)
        else:
            views = [ToolView(op_name="original", description="Original image.", image=base_image, metadata={})]

        serialized_views = []
        for view in views:
            if view.op_name != "original":
                user_content.append({"type": "text", "text": f"[Tool:{view.op_name}] {view.description}"})
            user_content.append({"type": "image"})
            image_inputs.append(view.image)
            serialized_views.append(
                {
                    "op_name": view.op_name,
                    "description": view.description,
                    "metadata": view.metadata,
                }
            )

        tool_log.append({"image_path": str(abs_path), "views": serialized_views})

    user_content.append({"type": "text", "text": format_question(sample)})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
        {"role": "user", "content": user_content},
    ]
    return messages, tool_log, image_inputs


def _serialize_choices(choices: List[Choice]) -> List[Dict[str, str]]:
    return [{"label": choice.label, "text": choice.text} for choice in choices]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    samples = load_mathvista_samples(args.dataset, limit=args.limit)
    logging.info("Loaded %d samples from %s", len(samples), args.dataset)

    toolbox = VisionToolbox(max_extra_views=args.tool_max_views) if args.use_tools else None
    if toolbox:
        logging.info("Vision toolbox enabled: %s", toolbox.describe())

    torch_dtype = _resolve_dtype(args.torch_dtype)
    logging.info("Loading model %s", args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model, "eval"):
        model.eval()
    processor = hf_processor(args.model, trust_remote_code=args.trust_remote_code)
    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    target_device = _infer_model_device(model)

    os.makedirs(args.output.parent, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    image_root = Path(args.image_root)
    rows: List[Dict] = []
    writer: Optional[pq.ParquetWriter] = None
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        try:
            messages, tool_log, image_inputs = _build_messages(
                sample, args.system_prompt, image_root, toolbox
            )
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[prompt],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            )
            inputs = inputs.to(target_device, dtype=torch_dtype)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            predicted_choice = extract_choice_from_response(response, sample.choices)
        except Exception as exc:
            logging.exception("Failed on sample %s: %s", sample.question_id, exc)
            response = f"__ERROR__: {exc}"
            predicted_choice = None
            tool_log = []

        row = {
            "question_id": sample.question_id,
            "data_source": sample.data_source,
            "question": sample.question,
            "answer": sample.answer,
            "choices": json.dumps(_serialize_choices(sample.choices)),
            "image_paths": json.dumps(sample.image_paths),
            "response": response,
            "prediction": predicted_choice,
            "tool_calls": json.dumps(tool_log),
            "model_name": args.model,
            "system_prompt": args.system_prompt,
            "use_tools": bool(toolbox),
        }
        rows.append(row)

        if args.save_every and len(rows) % args.save_every == 0:
            writer = _flush(rows, args.output, writer)

    writer = _flush(rows, args.output, writer)
    if writer is not None:
        writer.close()
    logging.info("Saved %s", args.output)


def _flush(rows: List[Dict], out_path: Path, writer: Optional[pq.ParquetWriter]) -> Optional[pq.ParquetWriter]:
    if not rows:
        return writer
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    rows.clear()
    return writer


if __name__ == "__main__":
    main()
