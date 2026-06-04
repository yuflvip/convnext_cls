from cls_engine.distributed.ddp import is_main_process


def rank_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)
