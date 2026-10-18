"""Qwen3-VL single-GPU, single-turn inference for ReferFlow-Bench."""
import argparse
import json
from pathlib import Path
from referflow_bench.io import read_jsonl, index_rows, write_json, sha256, image_file
from referflow_bench.protocol import build_prompt, validate_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prediction_file", default="results/qwen3vl.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Token limit and sample limit must be positive")
    rows = read_jsonl(args.annotation_file)
    index_rows(rows)
    validate_annotations(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    output = Path(args.prediction_file)
    if output.exists():
        parser.error("Prediction file already exists; choose a new path")
    for row in rows:
        image_file(args.image_dir, row["image_path"])
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=args.revision, dtype="auto", device_map="cuda:0").eval()
    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(image_file(args.image_dir, row["image_path"]))},
                {"type": "text", "text": build_prompt(row, "1000")},
            ]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            stream.write(json.dumps({"sample_id": row["sample_id"], "coordinate_system": "1000",
                "response": response}, ensure_ascii=False) + "\n")
            stream.flush()
    write_json(output.with_suffix(".run.json"), {"model": args.model_path, "revision": args.revision,
        "annotations_sha256": sha256(args.annotation_file), "coordinate_system": "1000",
        "do_sample": False, "max_new_tokens": args.max_new_tokens, "sample_count": len(rows),
        "protocol": "single_turn_all_steps"})


if __name__ == "__main__":
    main()
