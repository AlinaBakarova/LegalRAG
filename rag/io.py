from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def json_loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)


def read_json(path):
    return json_loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path, schema=None):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty JSONL line")
            try:
                row = json_loads(line)
                row = schema.model_validate(row) if schema else row
                key = getattr(row, "id", None) or getattr(row, "block_id", None)
                if schema and key is not None:
                    if key in seen:
                        raise ValueError(f"Duplicate ID: {key}")
                    seen.add(key)
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def dump_line(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(dump_line(row))
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

