import json
import re
from .metrics import valid_box

COORDINATES = ("pixel", "normalized", "1000")
TASKS = {
    "refinement": ("attribute", "relational"),
    "shift": ("spatial", "semantic"),
    "reset": ("cross_category", "intra_category"),
    "mixture": ("sequential", "interleaved"),
}


def validate_annotations(rows):
    if not rows:
        raise ValueError("Annotation file is empty")
    for row in rows:
        key = row["sample_id"]
        w, h = row.get("width"), row.get("height")
        if any(type(v) is not int or v <= 0 for v in (w, h)):
            raise ValueError(f"{key}: invalid image dimensions")
        if row.get("task_type") not in TASKS or row.get("sub_task_type") not in TASKS[row["task_type"]]:
            raise ValueError(f"{key}: invalid task/subtype")
        steps = row.get("steps")
        if not isinstance(steps, list) or not steps or row.get("sequence_length") != len(steps):
            raise ValueError(f"{key}: invalid sequence length")
        for i, step in enumerate(steps, 1):
            boxes = step.get("boxes")
            if step.get("step_id") != i or not isinstance(step.get("query"), str) or not step["query"].strip():
                raise ValueError(f"{key}: invalid step {i}")
            if not isinstance(boxes, list) or not boxes or not all(valid_box(b) for b in boxes):
                raise ValueError(f"{key}: invalid target boxes at step {i}")
            if any(not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h) for b in boxes):
                raise ValueError(f"{key}: target outside image at step {i}")


def build_prompt(row, coordinates="1000"):
    if coordinates not in COORDINATES:
        raise ValueError("Unknown coordinate system")
    units = {"pixel": f"pixel coordinates in the original {row['width']} by {row['height']} image",
             "normalized": "relative image coordinates from 0 to 1",
             "1000": "relative image coordinates from 0 to 1000"}[coordinates]
    # Only the step queries, their order, and image dimensions reach the model.
    return ("Locate all target regions for each ordered grounding step in the image.\n"
            "Resolve references using earlier steps when needed. Return all steps in one response.\n"
            f"Use {units}; boxes must be [x_min, y_min, x_max, y_max].\n"
            'Return only JSON: {"steps": [{"boxes": [[x_min, y_min, x_max, y_max], ...]}, ...]}.\n'
            "Include every matching target. Use an empty boxes array if no target is found.\n"
            "Grounding steps:\n" + "\n".join(f"{i}. {step['query']}" for i, step in enumerate(row["steps"], 1)))


def parse_response(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence[1]
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value.get("steps") if isinstance(value, dict) else None


def prediction_steps(prediction, row, default_coordinates=None):
    if prediction is None or prediction.get("error"):
        return [], "missing_or_failed"
    coordinates = prediction.get("coordinate_system", default_coordinates)
    if coordinates not in COORDINATES:
        raise ValueError(f"{row['sample_id']}: declare coordinate_system as pixel, normalized, or 1000")
    steps = prediction.get("steps") if "steps" in prediction else parse_response(prediction.get("response"))
    if not isinstance(steps, list):
        return [], "unparseable"
    sx, sy = {"pixel": (1, 1), "normalized": (row["width"], row["height"]),
              "1000": (row["width"] / 1000, row["height"] / 1000)}[coordinates]
    converted = []
    for step in steps:
        boxes = step.get("boxes") if isinstance(step, dict) else None
        if not isinstance(boxes, list) or not all(valid_box(b) for b in boxes):
            converted.append(None)
        else:
            converted.append([[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes])
    return converted, "parsed"
