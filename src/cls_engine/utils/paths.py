from pathlib import Path
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_default_run_name(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("exp_%Y%m%d%H%M%S")


def _timestamp_suffix(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d%H%M%S")


def resolve_output_dir(
    project: str,
    name: str,
    output: str | Path | None,
    exist_ok: bool,
    root: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[Path, str]:
    base_root = Path(root) if root is not None else Path.cwd()
    if output is not None:
        path = Path(output)
        if not path.is_absolute():
            path = base_root / path
        if path.exists() and not exist_ok:
            raise ValueError(f"Output path already exists: {path}")
        return path, name

    base_dir = base_root / "runs" / "classify" / project
    path = base_dir / name
    final_name = name
    if path.exists() and not exist_ok:
        suffix = _timestamp_suffix(now)
        final_name = f"{name}_{suffix}"
        path = base_dir / final_name
        while path.exists():
            suffix = _timestamp_suffix()
            final_name = f"{name}_{suffix}"
            path = base_dir / final_name
    return path, final_name
