import json
from pathlib import Path
from typing import Any


def _csv_cell(value: Any) -> str:
    text = str(value)
    if "," in text or '"' in text:
        text = '"' + text.replace('"', '""') + '"'
    return text


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(_csv_cell(item) for item in row) + "\n")


def append_csv_row(path: Path, row: list[Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(",".join(_csv_cell(item) for item in row) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
