from pathlib import Path
from typing import Any

from cls_engine.metrics.reports import confusion_to_report

from .writers import append_csv_row, write_csv, write_json, write_text


class ArtifactWriter:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_classes(self, class_names: list[str]) -> None:
        write_json(
            self.out_dir / "classes.json",
            {"classes": class_names, "class_to_idx": {name: i for i, name in enumerate(class_names)}},
        )

    def write_run_config(self, payload: dict[str, Any]) -> None:
        write_json(self.out_dir / "run_config.json", payload)

    def write_split_indices(self, payload: dict[str, Any]) -> None:
        write_json(self.out_dir / "split_indices.json", payload)

    def write_eval_outputs(self, prefix: str, class_names: list[str], eval_out: dict[str, Any]) -> None:
        all_rows = eval_out.get("all_rows", [])
        wrong_rows = sorted(eval_out.get("wrong_rows", []), key=lambda row: float(row[-1]), reverse=True)
        cm = eval_out.get("confusion_matrix")
        write_csv(
            self.out_dir / f"{prefix}_preds.csv",
            ["path", "gt_idx", "gt_name", "pred_idx", "pred_name", "conf"],
            all_rows,
        )
        write_csv(
            self.out_dir / f"wrong_{prefix}.csv",
            ["path", "gt_idx", "gt_name", "pred_idx", "pred_name", "conf"],
            wrong_rows,
        )
        if cm is not None:
            cm_rows = [[gt_name] + cm[i].tolist() for i, gt_name in enumerate(class_names)]
            write_csv(self.out_dir / f"confusion_matrix_{prefix}.csv", ["gt\\pred"] + class_names, cm_rows)
            write_text(self.out_dir / f"classification_report_{prefix}.txt", confusion_to_report(cm, class_names))

    def append_epoch_result(self, row: list[Any], header: list[str], first_epoch: bool) -> None:
        path = self.out_dir / "results.csv"
        if first_epoch:
            write_csv(path, header, [row])
        else:
            append_csv_row(path, row)

    def write_final_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.out_dir / "final_summary.json", payload)
