from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import socket
import time


DEFAULT_TORCH_PORT = 29500
STALE_LOCK_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class PortConfig:
    master_port: int | str = "auto"
    range_start: int = 20000
    range_end: int = 65000
    lock_dir: Path = Path("/tmp/cls_engine_ports")


def derive_master_port(
    data_root: str,
    model_name: str,
    output_dir: str,
    run_id: str,
    range_start: int,
    range_end: int,
) -> int:
    key = f"{data_root}|{model_name}|{output_dir}|{run_id}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    span = range_end - range_start
    return range_start + (int(digest[:8], 16) % (span + 1))


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _lock_path(lock_dir: Path, port: int) -> Path:
    return Path(lock_dir) / f"{port}.lock"


def _try_lock(lock_dir: Path, port: int) -> Path | None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(lock_dir, port)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if _is_stale_lock(path) and is_port_available(port):
            path.unlink(missing_ok=True)
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\ncreated={int(time.time())}\n")
    return path


def _is_stale_lock(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > STALE_LOCK_SECONDS


def _set_env_and_info(mode: str, port: int, config: PortConfig, lock_path: Path | None) -> tuple[int, dict]:
    os.environ["MASTER_PORT"] = str(port)
    return port, {
        "mode": mode,
        "port": port,
        "range_start": config.range_start,
        "range_end": config.range_end,
        "lock_path": str(lock_path) if lock_path is not None else None,
    }


def resolve_master_port(
    config: PortConfig,
    data_root: str,
    model_name: str,
    output_dir: str,
    run_id: str,
) -> tuple[int, dict]:
    env_port = os.environ.get("MASTER_PORT", "")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and env_port:
        return _set_env_and_info("torchrun_env", int(env_port), config, None)

    if isinstance(config.master_port, int) and config.master_port > 0:
        if not is_port_available(config.master_port):
            raise RuntimeError(f"Explicit MASTER_PORT is unavailable: {config.master_port}")
        return _set_env_and_info("explicit", config.master_port, config, None)

    if env_port and env_port != str(DEFAULT_TORCH_PORT):
        port = int(env_port)
        return _set_env_and_info("env", port, config, None)

    start = int(config.range_start)
    end = int(config.range_end)
    first = derive_master_port(data_root, model_name, output_dir, run_id, start, end)
    span = end - start + 1
    for offset in range(span):
        port = start + ((first - start + offset) % span)
        lock_path = _try_lock(config.lock_dir, port)
        if lock_path is None:
            continue
        if is_port_available(port):
            return _set_env_and_info("auto", port, config, lock_path)
        lock_path.unlink(missing_ok=True)

    raise RuntimeError(f"No available MASTER_PORT in range {start}-{end}")
