from cls_engine.distributed.ddp import is_main_process


def write_eval_artifacts(writer, prefix: str, eval_out: dict, class_names: list[str], print_top_wrong: int) -> None:
    if not is_main_process():
        return
    writer.write_eval_outputs(prefix, class_names, eval_out)
    wrong_rows = sorted(eval_out.get("wrong_rows", []), key=lambda row: float(row[-1]), reverse=True)
    all_rows = eval_out.get("all_rows", [])
    print(f"{prefix.capitalize()} wrong samples: {len(wrong_rows)} / {len(all_rows)}")
    if wrong_rows and print_top_wrong > 0:
        top_n = min(print_top_wrong, len(wrong_rows))
        print(f"Top-{top_n} most confident wrong ({prefix}):")
        for row in wrong_rows[:top_n]:
            print("  ", row)
