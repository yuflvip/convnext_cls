# Image Classification

Production-style image classification training project.

The first implemented command is `tools/train.py`. `tools/export.py` and
`tools/predict.py` are reserved entry points for later phases.

## Train

```bash
python tools/train.py \
  --config configs/cls_default.yaml \
  --data /path/to/dataset \
  --project vehicle_cls \
  --name baseline \
  --model convnext_nano.in12k_ft_in1k
```

The `--data` argument accepts either a single dataset root or a comma-separated
list of dataset roots:

```bash
python tools/train.py --data /data/side
python tools/train.py --data /data/side,/data/front
python tools/train.py --data /data/side,
```

Each dataset root must contain `train/` and may contain explicit `val/` and
`test/` directories:

```text
dataset_root/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

If `val/` is absent, the trainer creates a stratified split from the merged
`train/` samples across all dataset roots.

When multiple dataset roots are provided:

- Every root must contain `train/`
- Every root must expose the same class directory set
- `val/` and `test/` must be consistent across roots: either every root has the
  split or none of them do

## Config Inheritance

`configs/cls_default.yaml` can be used as the default base config. Custom
model configs can inherit from it with `_base_`, and child values override the
parent while unspecified fields continue to use the parent values.

```yaml
_base_: cls_default.yaml

model:
  name: hgnetv2_b4.ssld_stage2_ft_in1k

train:
  batch_size: 64
  lr: 0.0001
```

The final priority order is: base config < child config < CLI arguments.

## DDP

For external `torchrun`, pass an explicit master port when running several jobs
on the same machine:

```bash
torchrun --nproc_per_node=2 --master_port=29611 tools/train.py \
  --config configs/cls_default.yaml \
  --data /path/to/dataset \
  --project vehicle_cls \
  --name ddp_run
```

The internal distributed setup also supports `distributed.master_port: auto`.
Auto mode avoids the default `29500`, checks whether the port is bindable, and
uses a lock file under `/tmp/cls_engine_ports/` to reduce collisions between
concurrent training jobs.

## Devices

Training uses CUDA automatically when available unless `--device cpu` is passed.
CPU handles data discovery, image loading, split creation, and artifact writing.
GPU handles model forward/backward, AMP, optimizer steps, evaluation forward,
and DDP tensor communication.

If `--output` is omitted, the trainer writes to
`runs/classify/{project}/{name}`. When that default directory already exists
and `--exist_ok` is not set, the trainer appends `_{YYYYMMDDHHmmss}` to the
final run name to avoid collisions.
