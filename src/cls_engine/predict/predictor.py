import json
import random
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path
from cls_engine.io.writers import write_csv, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_predict_imgsz(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError("imgsz must be positive.")
        return (size, size)
    if len(parts) == 2:
        height = int(parts[0])
        width = int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz must be positive.")
        return (height, width)
    raise ValueError("imgsz must be a single integer or h,w.")


def collect_input_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def is_url_input(value: str | Path) -> bool:
    text = str(value)
    return text.startswith("http://") or text.startswith("https://")


def _build_temp_download_path(temp_dir: str | Path, suffix: str = ".jpg", now: datetime | None = None) -> Path:
    current = now or datetime.now()
    millis = int(current.microsecond / 1000)
    name = f"{current.strftime('%Y%m%d%H%M%S')}{millis:03d}_{random.randint(0, 999999):06d}{suffix}"
    return Path(temp_dir) / name


def prepare_prediction_inputs(data: str | Path, temp_dir: str | Path) -> tuple[list[Path], list[Path]]:
    if not is_url_input(data):
        return collect_input_images(data), []

    suffix = Path(str(data)).suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = ".jpg"
    temp_path = _build_temp_download_path(temp_dir, suffix=suffix)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(str(data)) as response:
        temp_path.write_bytes(response.read())
    return [temp_path], [temp_path]


def cleanup_temporary_inputs(temp_paths: list[Path], temp_dir: str | Path | None = None) -> None:
    for path in temp_paths:
        Path(path).unlink(missing_ok=True)
    if temp_dir is not None:
        temp_dir_path = Path(temp_dir)
        try:
            if temp_dir_path.is_dir() and not any(temp_dir_path.iterdir()):
                temp_dir_path.rmdir()
        except OSError:
            pass


def resolve_prediction_output_dir(
    model_path: str | Path,
    output: str | Path | None,
    now: datetime | None = None,
) -> Path:
    if output:
        return Path(output)
    current = now or datetime.now()
    return Path("runs") / "predict" / f"predict_{current.strftime('%Y%m%d%H%M%S')}"

def load_onnx_classes(model_path: str | Path, classes_path: str | Path | None) -> list[str]:
    resolved = Path(classes_path) if classes_path else Path(model_path).with_name("classes.json")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    classes = payload.get("classes") or []
    if not classes:
        raise ValueError(f"Classes file does not contain classes: {resolved}")
    return classes


def write_prediction_outputs(output_dir: str | Path, rows: list[dict]) -> None:
    output_dir = Path(output_dir)
    csv_rows = []
    for row in rows:
        csv_rows.append([
            row["path"],
            row["pred_idx"],
            row["pred_name"],
            f'{row["conf"]:.6f}',
            json.dumps(row["topk"], ensure_ascii=False),
        ])
    write_csv(output_dir / "predictions.csv", ["path", "pred_idx", "pred_name", "conf", "topk"], csv_rows)
    write_json(output_dir / "predictions.json", rows)


def print_prediction_progress(index: int, total: int, row: dict) -> None:
    print(
        f"[Predict] {index}/{total} {row['path']} -> "
        f"class={row['pred_name']} id={row['pred_idx']} conf={row['conf']:.4f}"
    )


def arrange_prediction_outputs(output_dir: str | Path, rows: list[dict], arrange_mode: str | None) -> None:
    if not arrange_mode:
        return

    output_dir = Path(output_dir)
    for row in rows:
        source = Path(row.get("local_path", row["path"]))
        target = output_dir / str(row["pred_name"]) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if arrange_mode == "copy":
            shutil.copy2(source, target)
        elif arrange_mode == "move":
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
        else:
            raise ValueError(f"Unsupported arrange mode: {arrange_mode}")
