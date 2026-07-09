# 分类器 Loss 优化方案与实现设计

## 1. 目标

本文档用于设计当前分类训练框架中的 loss 优化路线，目标是：

- 提升闭集分类准确率
- 提升跨场景泛化稳定性
- 改善类别不平衡和易混类别表现
- 在不破坏当前训练主流程的前提下，逐步扩展 loss 能力

本文档覆盖两类路线：

1. 当前闭集分类主线可直接落地的 loss
2. 更偏表征学习和 teacher-student 训练的后续 loss


## 2. 当前代码现状

当前训练代码的 loss 设计非常简单：

- 训练主损失只有 `nn.CrossEntropyLoss(weight=class_weights)`，见 `src/cls_engine/engine/trainer.py`
- 类别不平衡通过 `class_weight_mode` 生成类别权重，见 `src/cls_engine/data/splits.py`
- 训练循环只接受单个 `criterion(logits, y)`，见 `src/cls_engine/engine/loops.py`
- 验证阶段使用普通 `F.cross_entropy(..., reduction="sum")` 计算平均 `val_loss`

当前没有：

- focal loss
- label smoothing
- triplet loss
- center loss
- contrastive loss
- distillation loss
- 多 loss 加权组合

这意味着当前框架适合先扩成：

- 单主损失可切换
- 主损失 + 辅助损失组合
- 可选 teacher 路线


## 3. 问题分类

从业务现象看，loss 优化主要解决的是下面几类问题：

### 3.1 基础闭集分类准确率不足

表现：

- 训练能收敛，但最终 top1 不高
- 易混类边界不够清晰

优先方案：

- CrossEntropy
- Label Smoothing
- Focal Loss

### 3.2 泛化能力不足

表现：

- 同场景验证集效果好
- 新场景掉点明显
- 预测过度自信

优先方案：

- Label Smoothing
- Distillation
- Contrastive 预训练或联合训练

说明：

泛化问题通常优先由数据、预处理、增强决定，loss 只应作为第二层优化。

### 3.3 类别不平衡和难样本问题

表现：

- 少数类 recall 明显偏低
- 容易类压制难类
- 固定几类之间混淆严重

优先方案：

- 带 class weights 的 CrossEntropy
- Focal Loss

### 3.4 特征分布不理想

表现：

- 类内差异大
- 类间边界粘连
- 想做检索、聚类、开集扩展

优先方案：

- Center Loss
- Triplet Loss
- Contrastive Loss


## 4. 候选 Loss 的作用与推荐级别

### 4.1 CrossEntropyLoss

作用：

- 标准闭集分类损失
- 直接优化类别概率

优点：

- 稳定
- 简单
- 与当前代码完全兼容

缺点：

- 对难样本关注不够
- 对过度自信没有约束

推荐级别：

- 必保留
- 作为所有第一阶段方案的基线

### 4.2 Label Smoothing

作用：

- 把硬标签从 one-hot 变成轻微软标签
- 降低模型过度自信

适用问题：

- 训练集高分，跨场景变差
- 标签边界并非绝对干净
- 模型输出置信度过高

优点：

- 实现简单
- 风险低
- 往往能改善泛化与校准

缺点：

- 对极限训练精度可能略有压制

推荐级别：

- 第一优先级

推荐参数：

- `label_smoothing = 0.05`
- 可对比 `0.0 / 0.05 / 0.1`

### 4.3 Focal Loss

作用：

- 降低易分类样本权重
- 强化难样本和易混样本的学习

适用问题：

- 类别不平衡
- 少数类表现差
- 特定类别长期被吞掉

优点：

- 对困难样本更敏感
- 对细粒度混淆较有效

缺点：

- 增加参数复杂度
- 如果数据本身不难，可能不如 CrossEntropy 稳

推荐级别：

- 第一优先级
- 与 Label Smoothing 并列，但建议晚于 Label Smoothing 接入

推荐参数：

- `gamma = 1.5 ~ 2.0`
- `alpha` 先不暴露，优先复用现有 class weights

### 4.4 Center Loss

作用：

- 约束同类特征向类中心聚拢
- 降低类内方差

适用问题：

- 类内差异大
- 细粒度结构类任务

优点：

- 对特征空间聚合有效

缺点：

- 需要额外维护类中心
- 需要模型暴露 embedding
- 训练复杂度高于 CE/Focal

推荐级别：

- 第二优先级

推荐参数：

- `center_loss_weight = 0.01 ~ 0.1`

### 4.5 Triplet Loss

作用：

- 拉近同类样本
- 推远异类样本

适用问题：

- 度量学习
- 检索
- 细粒度类别聚类

优点：

- 可改善 embedding 结构

缺点：

- 依赖采样策略
- 对普通闭集分类不是最直接的提效手段

推荐级别：

- 第二优先级偏后

推荐参数：

- `margin = 0.2 ~ 0.5`

### 4.6 Contrastive Loss

作用：

- 学习样本间相似与不相似关系
- 更侧重表征学习

适用问题：

- 自监督或半监督
- 跨域泛化
- 后续想扩到检索或相似度学习

优点：

- 对特征泛化潜力大

缺点：

- 对当前训练链路改动大
- 通常需要双视图增强或样本对组织

推荐级别：

- 第三优先级

### 4.7 Distillation Loss

作用：

- 用 teacher 模型指导 student 模型
- 提升 student 精度和泛化

适用问题：

- 已有更强 teacher
- 想部署更小模型
- 想让 student 学会 teacher 的类别分布

优点：

- 部署价值高
- 对小模型比较有效

缺点：

- 需要 teacher checkpoint
- 增加训练链路复杂度

推荐级别：

- 第一优先级中的后置方案
- 应在 CE/LS/Focal 稳定后再做

推荐参数：

- `distill_weight = 0.25 ~ 0.5`
- `temperature = 2.0 ~ 4.0`


## 5. 推荐优先级

### 5.1 第一阶段

目标：

- 最小改动
- 快速验证能否提高准确率与泛化

建议顺序：

1. `CrossEntropy + class weights` 保持不变，作为基线
2. `CrossEntropy + Label Smoothing`
3. `Focal Loss`
4. `Focal Loss + class weights`

### 5.2 第二阶段

目标：

- 进一步改善特征结构

建议顺序：

1. `CrossEntropy + Center Loss`
2. `CrossEntropy + Triplet Loss`

### 5.3 第三阶段

目标：

- 做 teacher-student 或表示学习增强

建议顺序：

1. `CrossEntropy + Distillation`
2. `CrossEntropy + Contrastive` 或先做对比预训练


## 6. 建议的配置设计

当前 `TrainSettings` 已包含 `class_weight_mode`、`lr`、`weight_decay` 等项。loss 扩展建议集中在 `train` 下，保持单入口。

建议新增字段：

```yaml
train:
  loss:
    name: ce
    label_smoothing: 0.0
    focal_gamma: 2.0
    focal_alpha: null
    center_weight: 0.0
    triplet_weight: 0.0
    triplet_margin: 0.3
    contrastive_weight: 0.0
    distill_weight: 0.0
    distill_temperature: 2.0
    teacher_model: ""
```

其中：

- `name` 可选：`ce`、`focal`
- `label_smoothing` 仅对 CE 主损失生效
- `center_weight / triplet_weight / contrastive_weight / distill_weight` 默认 `0.0` 表示关闭

推荐不做的设计：

- 用多个互相冲突的布尔开关，如 `use_focal`, `use_triplet`, `use_distill`
- 让一个 loss 的启用方式散落在多个配置块


## 7. 实现设计

### 7.1 总体原则

实现应满足：

- 不破坏当前 `train_one_epoch()` 的主循环接口
- 先兼容单损失，再扩展到组合损失
- 验证阶段仍保留统一 `val_loss` 口径
- 配置未启用时，行为与当前版本一致

### 7.2 模块拆分

建议新增：

- `src/cls_engine/losses/__init__.py`
- `src/cls_engine/losses/focal.py`
- `src/cls_engine/losses/center.py`
- `src/cls_engine/losses/triplet.py`
- `src/cls_engine/losses/contrastive.py`
- `src/cls_engine/losses/distill.py`
- `src/cls_engine/losses/factory.py`

其中：

- `factory.py` 负责根据配置构建 `criterion`
- 每个 loss 文件只负责单一损失定义

### 7.3 Criterion 接口设计

当前训练循环假设：

```python
loss = criterion(logits, y)
```

为了支持 embedding 和 teacher loss，建议逐步收敛成：

```python
loss_dict = criterion(
    logits=logits,
    targets=y,
    features=features,
    teacher_logits=teacher_logits,
)
```

返回格式建议：

```python
{
  "loss": total_loss,
  "loss_ce": ...,
  "loss_focal": ...,
  "loss_center": ...,
  "loss_triplet": ...,
  "loss_contrastive": ...,
  "loss_distill": ...,
}
```

但为了控制第一阶段改动，建议分两步：

#### 第一步

只支持：

- `criterion(logits, y) -> tensor`

即先落地：

- CE
- CE + label smoothing
- Focal

#### 第二步

把训练循环改成兼容 dict 输出：

- 若 `criterion(...)` 返回 tensor，则沿用旧逻辑
- 若返回 dict，则取 `loss_dict["loss"]`

这样能平滑支持：

- Center
- Triplet
- Distillation

### 7.4 模型输出设计

当前 `build_model()` 只返回分类 logits。  
若要支持 Center / Triplet / Contrastive，模型需要暴露 embedding。

建议新增两种方案中的一种：

#### 方案 A：包装模型输出

新增分类器包装器，返回：

```python
{
  "logits": logits,
  "features": embedding,
}
```

优点：

- 扩展性更好

缺点：

- 对当前训练和预测路径改动更大

#### 方案 B：仅在需要时挂一个投影头

保持默认模型输出 logits，只有当 metric learning 类 loss 打开时，模型返回 `(logits, features)`。

优点：

- 对现有逻辑侵入较小

建议：

- 第一阶段不改
- 第二阶段采用方案 B

### 7.5 Distillation 设计

distillation 需要：

- 加载 teacher checkpoint
- teacher 固定为 `eval()` 且不反传
- 同批输入同时过 student 和 teacher

总损失建议：

```text
loss = (1 - distill_weight) * loss_ce + distill_weight * loss_kd
```

其中：

- `loss_ce` 是 student 对真实标签的监督
- `loss_kd` 是 student logits 与 teacher logits 的 KL 散度

teacher 路线第一版应限制为：

- teacher 仅支持 `pth`
- teacher 类别空间必须与 student 一致


## 8. 验证与日志设计

### 8.1 训练日志

若启用多 loss，建议在训练输出中记录分项损失均值，例如：

```text
[Epoch 12] train_loss=0.7421 loss_ce=0.6812 loss_center=0.0609 ...
```

### 8.2 results.csv

建议新增可选列：

- `loss_ce`
- `loss_focal`
- `loss_center`
- `loss_triplet`
- `loss_contrastive`
- `loss_distill`

### 8.3 final_summary.json

建议写入：

- `loss_name`
- `label_smoothing`
- `focal_gamma`
- `center_weight`
- `triplet_weight`
- `contrastive_weight`
- `distill_weight`
- `teacher_model`


## 9. 针对车型分类器的推荐路线

结合当前任务特点，loss 路线建议如下：

### 9.1 第一优先级

- `CrossEntropy + label smoothing`
- `Focal Loss`

原因：

- 对现有项目改动小
- 能直接覆盖泛化不足、难样本、易混类别等问题

### 9.2 第二优先级

- `CrossEntropy + Distillation`

前提：

- 已经有一个更强的 teacher

### 9.3 暂缓项

- Center Loss
- Triplet Loss
- Contrastive Loss

原因：

- 当前更大的瓶颈通常不是特征损失本身
- 而是输入尺寸、预处理、增强和场景分布


## 10. 实验顺序建议

建议只做小步可解释实验，不同时引入多个新因素。

### Phase 0

基线：

- 当前 CE + class weights

### Phase 1

对比：

1. CE
2. CE + label smoothing=0.05
3. CE + label smoothing=0.1

观察：

- val_acc_top1
- 新场景验证集 acc
- 易混类别 recall

### Phase 2

对比：

1. Focal gamma=1.5
2. Focal gamma=2.0
3. Focal + class weights

观察：

- 少数类 recall
- 易混类 confusion

### Phase 3

若已有强 teacher：

1. CE + distillation
2. CE + label smoothing + distillation

### Phase 4

若前几轮都已稳定，且确实要做特征结构优化，再考虑：

- Center
- Triplet
- Contrastive


## 11. 风险与边界

### 11.1 不要把 loss 当成主要泛化手段

若模型跨场景掉点来自：

- 输入尺寸不够
- `stretch` 破坏比例
- 数据裁剪质量不稳定
- 验证集场景不独立

则仅靠换 loss 通常收益有限。

### 11.2 Focal 不一定总优于 CrossEntropy

在数据较干净、类别不算太难时，Focal 可能不如 CE 稳。

### 11.3 Metric learning 类 loss 改动更大

Center / Triplet / Contrastive 都要求：

- 模型暴露 embedding
- 训练循环支持多输入和多分项 loss

因此不应作为第一步改动。

### 11.4 Distillation 依赖 teacher 质量

如果 teacher 本身不稳，蒸馏通常不会带来正收益。


## 12. 最终建议

在当前项目中，loss 优化建议按下面顺序推进：

1. 保留 CE 基线
2. 先接入 Label Smoothing
3. 再接入 Focal Loss
4. 若存在更强 teacher，再接 Distillation
5. Center / Triplet / Contrastive 作为后续扩展，不作为第一阶段核心方案

从工程性价比看：

- `Label Smoothing` 是最低风险、最值得先落地的方案
- `Focal Loss` 是处理不平衡与易混类的第一优先增强
- `Distillation` 是 teacher 已成熟后的高价值方案
- `Triplet / Center / Contrastive` 适合作为第二阶段特征学习扩展
