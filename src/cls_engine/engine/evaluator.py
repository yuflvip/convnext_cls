from cls_engine.distributed.ddp import is_main_process


def write_eval_artifacts(writer, prefix: str, eval_out: dict, class_names: list[str], print_top_wrong: int) -> None:
    if not is_main_process():
        return
    writer.write_eval_outputs(prefix, class_names, eval_out)
    print(
        f"{prefix.capitalize()} metrics: macro_f1={eval_out.get('macro_f1', 0.0):.4f} "
        f"balanced_acc={eval_out.get('balanced_accuracy', 0.0):.4f} "
        f"worst_recall={eval_out.get('worst_class_recall', 0.0):.4f} "
        f"ece={eval_out.get('ece', 0.0):.4f} brier={eval_out.get('brier_score', 0.0):.4f} "
        f"nll={eval_out.get('nll', eval_out.get('val_loss', 0.0)):.4f}"
    )
    wrong_rows = sorted(eval_out.get("wrong_rows", []), key=lambda row: float(row[-1]), reverse=True)
    all_rows = eval_out.get("all_rows", [])
    print(f"{prefix.capitalize()} wrong samples: {len(wrong_rows)} / {len(all_rows)}")
    if wrong_rows and print_top_wrong > 0:
        top_n = min(print_top_wrong, len(wrong_rows))
        print(f"Top-{top_n} most confident wrong ({prefix}):")
        for row in wrong_rows[:top_n]:
            print("  ", row)
