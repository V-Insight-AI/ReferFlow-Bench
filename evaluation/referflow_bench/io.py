import hashlib
import json
from pathlib import Path


def image_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Image missing or outside data directory: {name}")
    return path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    return rows


def index_rows(rows):
    result = {}
    for row in rows:
        key = row.get("sample_id")
        if not isinstance(key, str) or not key:
            raise ValueError("Every row requires a nonempty sample_id")
        if key in result:
            raise ValueError(f"Duplicate sample_id: {key}")
        result[key] = row
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
