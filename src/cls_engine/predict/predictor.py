import json
from pathlib import Path
from cls_engine.io.writers import write_csv, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".