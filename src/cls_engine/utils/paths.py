from pathlib import Path
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, e