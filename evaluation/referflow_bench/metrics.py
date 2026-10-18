"""Exact, order-invariant box-set matching; no scipy or model dependencies."""
import math


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in box)
            and box[0] < box[2] and box[1] < box[3])


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def perfect_matching(edges, n):
    """Find a full bipartite matching with augmenting paths (not greedy IoU)."""
    owners = [-1] * n

    def augment(left, seen):
        for right in edges[left]:
            if right in seen:
                continue
            seen.add(right)
            if owners[right] == -1 or augment(owners[right], seen):
                owners[right] = left
                return True
        return False

    return all(augment(left, set()) for left in range(n))


def step_correct(predicted, target, threshold=0.5):
    if not isinstance(predicted, list) or len(predicted) != len(target):
        return False
    if not all(valid_box(box) for box in predicted):
        return False
    edges = [[j for j, box in enumerate(target) if iou(pred, box) >= threshold] for pred in predicted]
    return perfect_matching(edges, len(target))
