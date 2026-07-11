# 图像分类训练项目

这是一个面向生产使用方式的图像分类训练项目。

当前已实现的主要入口是 `tools/train.py`。`tools/export.py` 和
`tools/predict.py` 作为后续阶段的预留入口。

## 训练

```bash
python tools/train.py \
  --config configs/cls_default.yaml \
  --data /path/to/dataset \
  --project vehicle_cls \
  --name baseline \
  --model convnext_nano.in12k_ft_in1k
```

`--data` 参数支持单个数据集根目录，也支持多个根目录用逗号分隔：

```bash
python tools/train.py --data /data/side
python tools/train.py --data /data/side,/data/front
python tools/train.py --data /data/side,
```

每个数据集根目录都必须包含 `train/`，也可以显式提供 `val/` 和
`test/` 目录：

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

如果没有提供 `val/`，训练器会从所有数据集根目录合并后的 `train/`
样本中自动生成分层验证集划分。

当传入多个数据集根目录时：

- 每个根目录都必须包含 `train/`
- 每个根目录下的类别目录集合必须一致
- `val/` 和 `test/` 的存在情况必须一致：要么每个根目录都有，要么都没有

## 配置继承

`configs/cls_default.yaml` 作为默认基类配置。其他模型配置文件可以通过
`_base_` 继承它；子配置中显式定义的字段会覆盖父配置，未定义的字段继续
沿用父配置。

```yaml
_base_: cls_default.yaml

model:
  name: hgnetv2_b4.ssld_stage2_ft_in1k

train:
  batch_size: 64
  lr: 0.0001
```

最终优先级顺序为：基础配置 < 子配置 < 命令行参数。

## DDP

如果使用外部 `torchrun`，并且同一台机器上同时运行多个任务，建议显式指定
`master_port`：

```bash
torchrun --nproc_per_node=2 --master_port=29611 tools/train.py \
  --config configs/cls_default.yaml \
  --data /path/to/dataset \
  --project vehicle_cls \
  --name ddp_run
```

内部的分布式启动同样支持 `distributed.master_port: auto`。自动模式会避开
默认的 `29500`，检测端口是否可绑定，并通过 `/tmp/cls_engine_ports/`
下的锁文件降低并发训练任务之间的端口冲突概率。

## 设备

训练时如果机器上可用 CUDA，默认会自动使用；只有显式传入 `--device cpu`
时才强制走 CPU。CPU 主要负责数据发现、图像加载、数据集划分和结果落盘；
GPU 主要负责模型前向/反向、AMP、优化器更新、评估前向以及 DDP 张量通信。

如果没有传入 `--output`，训练结果默认写入
`runs/classify/{project}/{name}`。如果该默认目录已经存在，且没有传入
`--exist_ok`，训练器会在最终运行名后自动追加 `_{YYYYMMDDHHmmss}`，
以避免目录冲突。
