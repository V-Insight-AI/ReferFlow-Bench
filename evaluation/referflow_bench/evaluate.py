import argparse
from collections import defaultdict
from . import __version__
from .io import read_jsonl, index_rows, sha256, write_json
from .metrics import step_correct
from .protocol import COORDINATES, prediction_steps, validate_annotations


def aggregate(items):
    n = len(items)
    return {"n": n, **{name: 100 * sum(x[name] for x in items) / n if n else None
                         for name in ("q_acc", "t_acc", "f_acc")}}


def evaluate(annotations, predictions, coordinates=None):
    targets, predicted = index_rows(annotations), index_rows(predictions)
    validate_annotations(annotations)
    unknown = set(predicted) - set(targets)
    if unknown:
        raise ValueError(f"Unknown prediction sample_id(s): {', '.join(sorted(unknown)[:5])}")
    details = []
    groups = {name: defaultdict(list) for name in ("task_type", "sub_task_type", "sequence_length")}
    for key, row in targets.items():
        steps, status = prediction_steps(predicted.get(key), row, coordinates)
        length = len(row["steps"])
        correct = [step_correct(steps[i], target["boxes"]) if i < len(steps) else False
                   for i, target in enumerate(row["steps"])]
        item = {"sample_id": key, "correct_steps": correct, "predicted_steps": len(steps),
                "expected_steps": length, "status": status, "q_acc": sum(correct) / length,
                "t_acc": int(all(correct) and len(steps) == length), "f_acc": int(correct[-1])}
        details.append(item)
        for name, bins in groups.items():
            group_key = f"{row['task_type']}/{row[name]}" if name == "sub_task_type" else str(row[name])
            bins[group_key].append(item)
    result = {"evaluator_version": __version__, "protocol": "single_turn_all_steps",
              "iou_threshold": 0.5, "score_unit": "percent", "aggregation": "trajectory_macro",
              "overall": aggregate(details), "coverage": {"expected": len(targets), "submitted": len(predicted),
              "missing": len(targets) - len(predicted), "unparseable_or_failed": sum(d["status"] != "parsed" for d in details)},
              **{f"by_{name}": {key: aggregate(items) for key, items in sorted(bins.items())}
                 for name, bins in groups.items()}}
    return result, details


def main():
    p = argparse.ArgumentParser(description="ReferFlow-Bench Q-Acc / T-Acc / F-Acc at IoU >= 0.5")
    p.add_argument("--annotation_file", "--annotations", dest="annotations", required=True)
    p.add_argument("--prediction_file", "--predictions", dest="predictions", required=True)
    p.add_argument("--output", default="results/metrics.json")
    p.add_argument("--details", help="Optional per-trajectory diagnostics JSON")
    p.add_argument("--coordinate-system", choices=COORDINATES, help="Fallback for rows without coordinate_system")
    p.add_argument("--require-complete", action="store_true", help="Reject incomplete files for leaderboard submission")
    args = p.parse_args()
    try:
        result, details = evaluate(read_jsonl(args.annotations), read_jsonl(args.predictions), args.coordinate_system)
        if args.require_complete and result["coverage"]["missing"]:
            raise ValueError("Incomplete prediction coverage")
    except ValueError as error:
        p.error(str(error))
    result["annotations_sha256"] = sha256(args.annotations)
    result["predictions_sha256"] = sha256(args.predictions)
    write_json(args.output, result)
    if args.details:
        write_json(args.details, details)
    values = result["overall"]
    print(f"N={values['n']}  Q-Acc={values['q_acc']:.4f}  T-Acc={values['t_acc']:.4f}  F-Acc={values['f_acc']:.4f}")
    print(f"Missing predictions: {result['coverage']['missing']}; scores saved to {args.output}")


if __name__ == "__main__":
    main()
