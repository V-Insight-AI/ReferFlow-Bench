# Evaluation

Run the commands below from `evaluation/`.

## Qwen3-VL single-GPU inference

Download the [dataset](https://huggingface.co/datasets/V-Insight/ReferFlow-Bench) and extract `images.zip`. Set `--image_dir` to the extracted `images/` directory. The script uses the first visible GPU.

```bash
pip install torch torchvision transformers accelerate pillow
python eval_qwen3vl.py --model_path Qwen/Qwen3-VL-2B-Instruct --annotation_file /path/to/annotation.jsonl --image_dir /path/to/images --prediction_file results/qwen3vl.jsonl
```

## Scoring

```bash
python get_prediction_Acc.py --annotation_file /path/to/annotation.jsonl --prediction_file results/qwen3vl.jsonl --output results/metrics.json
```

Reports **Q-Acc**, **T-Acc**, and **F-Acc** at IoU ≥ 0.5, with breakdowns by type, subtype, and trajectory length. Add `--require-complete` when preparing a leaderboard submission.

## Prediction format

For other models, save one JSON object per trajectory, using its `sample_id` and returning every step's boxes in order:

```json
{"sample_id":"rfbench-00001","coordinate_system":"pixel","steps":[{"boxes":[[10,20,100,200]]},{"boxes":[[30,40,60,80]]}]}
```

Boxes use `[x1, y1, x2, y2]`. Supported coordinates: `pixel`, `1000`, and `normalized` (0–1). The example illustrates the format; use the actual number of steps and predicted boxes. Qwen3-VL outputs can be passed directly to the scoring script.
