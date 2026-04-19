# TorchSense vs Anomalib 架构对比

TorchSense 参考 anomalib 的模块化设计，在保持架构一致性的同时做了领域适配（图像异常检测 → 多模态生理信号处理）。

## 模块对应关系

| 模块 | anomalib | torchsense | 状态 |
|------|----------|------------|------|
| CLI | `AnomalibCLI` | `TorchSenseCLI` | 已实现 |
| Engine | `Engine` | `Engine`（含 `train`, `export`, `from_config`） | 已实现 |
| 基类模块 | `AnomalibModule(ExportMixin, LightningModule, ABC)` | `TorchSenseModule(ExportMixin, LightningModule)` | 已实现 |
| 模型注册 | `__subclasses__()` 自动发现 | `@register_model` 装饰器显式注册 | 已实现 |
| Dataset 基类 | `AnomalibDataset` | `TorchSenseDataset`（含 `subset()`, `subject_ids`） | 已实现 |
| DataModule 基类 | `AnomalibDataModule` | `TorchSenseDataModule`（含 `ValSplitMode`, `TestSplitMode`） | 已实现 |
| 划分策略 | `ValSplitMode`(5种), `TestSplitMode`(3种) | `ValSplitMode`(5种), `TestSplitMode`(3种) | 已实现 |
| PreProcessor | `PreProcessor(nn.Module, Callback)` | `PreProcessor(nn.Module, Callback)` | 已实现 |
| PostProcessor | `PostProcessor(nn.Module, Callback)` | `PostProcessor` + `SignalPostProcessor` + `BPPostProcessor` | 已实现 |
| Evaluator | `Evaluator(nn.Module, Callback)` | `Evaluator(nn.Module, Callback)` | 已实现 |
| Metric 基类 | `AnomalibMetric`（fields + strict） | `TorchSenseMetric`（fields + strict + 子类化 `extract()`） | 已实现 |
| Visualizer | `Visualizer(nn.Module, Callback)` | `Visualizer` + `SignalVisualizer` + `BPVisualizer` | 已实现 |
| ExportMixin | torch / onnx / openvino | ONNX | 已实现 |
| `_resolve_component` | `bool \| Instance` 三态解析 | `bool \| Instance \| None` 三态解析 | 已实现 |
| `from_config` | 类方法 + jsonargparse | 类方法 + jsonargparse | 已实现 |
| `configure_callbacks` | 注入 4 组件 | 注入 4 组件 | 已实现 |
| `forward()` 管道 | pre → model → post | pre → model → post | 已实现 |

## 仍存差异

### 架构设计差异（有意的领域适配）

| 差异点 | anomalib | torchsense | 原因 |
|--------|----------|------------|------|
| 数据批次 | 类型化 `Batch` / `ImageBatch` (NamedTuple) | 普通 `dict[str, Tensor]` | 多模态结构差异大，dict 更灵活 |
| 模型注册 | `__subclasses__()` 自动发现 | `@register_model` 装饰器 | 更可控、避免意外注册 |
| 训练步骤 | 子类各自实现 `training_step` | 基类模板调用 `model_step()` + `_compute_loss()` | 减少子类重复代码 |
| 步骤返回值 | `batch.update(**predictions)` | `{**output, **batch}` dict 合并 | 配合 dict 批次设计 |
| 指标提取 | `AnomalibMetric` 从 Batch 属性 `getattr` | `TorchSenseMetric.extract()` 子类化覆盖 | 适配 dict + 支持复杂提取（如列切片） |
| 导出格式 | torch + ONNX + OpenVINO | 仅 ONNX | 当前阶段够用，按需扩展 |
| 预处理分工 | `PreProcessor` 做 Resize + Normalize | DataModule 做滤波/分段（CPU/NumPy），PreProcessor 做增强（GPU/Tensor） | 信号处理需 scipy 等 CPU-only 库 |

### anomalib 有但 torchsense 尚未实现的功能

| 功能 | anomalib 实现 | 重要程度 | 说明 |
|------|-------------|---------|------|
| Logger 适配 | Comet / MLflow / TensorBoard / W&B 封装 | 中 | 目前用 Lightning 默认 logger |
| CLI 扩展子命令 | `train` / `predict` / `export` / `install` / `benchmark` | 中 | 当前仅 Lightning 默认 `fit/validate/test/predict` |
| `get_datamodule()` 工厂 | 按名称/配置创建 DataModule | 中 | 数据集多了后需要 |
| Pipeline 系统 | Benchmark / Tiled Ensemble | 低 | 自动化评测管道 |
| Inferencer | `TorchInferencer` / `OpenVINOInferencer` | 低 | ONNX + onnxruntime 可替代 |
| Data Validators | `validators/` 输入校验 | 低 | 信号场景暂不急需 |
| NNCF 量化 | `NNCFCallback` | 低 | 部署时再加 |
| PyTorch / OpenVINO 导出 | `.pt` + `.xml/.bin` | 低 | ONNX 已够用 |
| `trainer_arguments` | 子类声明 Trainer 参数 | 低 | 当前模型简单 |
| `input_size` 属性 | 从 transform 链推断输入尺寸 | 低 | 图像场景关键，信号不太需要 |

## 基类模块对比

### 四个子组件（完全对齐）

| 组件 | anomalib | torchsense |
|------|----------|------------|
| `pre_processor` | `PreProcessor(nn.Module, Callback)` 操作 `batch.image` | `PreProcessor(nn.Module, Callback)` 操作 `batch["input"]` |
| `post_processor` | `PostProcessor(nn.Module, Callback)` 归一化/阈值 | `PostProcessor(nn.Module, Callback)` 信号 clamp / BP 校准 |
| `evaluator` | `Evaluator(nn.Module, Callback)` 管理 AUROC/F1 等 | `Evaluator(nn.Module, Callback)` 管理 MSE/PCC/MAE 等 |
| `visualizer` | `ImageVisualizer(Visualizer)` 异常图 | `SignalVisualizer` / `BPVisualizer` 信号对比图 |

### 组件配置（三态解析）

```python
# 两边都支持：
model = MyModel(evaluator=True)      # 使用默认
model = MyModel(evaluator=False)     # 禁用
model = MyModel(evaluator=custom)    # 自定义实例
```

### forward() 管道

```python
# 两边的 forward() 都是 export pipeline：
def forward(self, x):
    x = self.pre_processor(x) if self.pre_processor else x
    x = self.model(x)
    x = self.post_processor(x) if self.post_processor else x
    return x
```

### Training 步骤

| | anomalib | torchsense |
|--|----------|------------|
| 训练步骤 | 子类实现 `training_step` | 基类模板调用 `model_step()` + `_compute_loss()` |
| 模型推理 | `self.model(batch.image)` | `self.model_step(batch)` → 返回 output dict |
| 步骤返回值 | `batch.update(**predictions)` | `{**output, **batch}` 合并 dict |
| 指标管理 | 纯 Evaluator callback | 纯 Evaluator callback |

### 指标系统

| | anomalib | torchsense |
|--|----------|------------|
| 基类 | `AnomalibMetric`（Mixin + 具体 Metric 多继承） | `TorchSenseMetric`（包装器，内含 Metric 实例） |
| 字段提取 | `getattr(batch, field)` 从 dataclass 属性取值 | `step_output[field]` 从 dict 取值 |
| 复杂提取 | 不支持，字段必须预先存在 | 子类覆盖 `extract()` 方法（如列切片） |
| strict 模式 | 支持（`strict=True/False`） | 支持（`strict=True/False`） |

### Data 层

| | anomalib | torchsense |
|--|----------|------------|
| 层次 | `datasets/` 和 `datamodules/` 分离 | `datasets/` 和 `datamodules/` 分离 |
| Dataset 基类 | `AnomalibDataset`（基于 DataFrame） | `TorchSenseDataset`（`subset()` + `subject_ids`） |
| 划分策略 | `ValSplitMode`(5种), `TestSplitMode`(3种) | `ValSplitMode`(5种), `TestSplitMode`(3种) |
| 子类职责 | `_setup()` 加载数据 | `_setup()` 加载数据并创建 `_full_dataset` |
| 划分执行 | 基类统一处理 | 基类统一处理 |

## 设计模式对比

| 设计模式 | anomalib | torchsense |
|---------|----------|------------|
| 模板方法 | `AnomalibModule.forward` 定义管道 | `TorchSenseModule.forward` 定义管道 |
| 组合模式 | 4 组件（pre/post/evaluator/visualizer） | 4 组件（pre/post/evaluator/visualizer） |
| 工厂模式 | `get_model()`, `get_datamodule()` | `get_model()` |
| 策略模式 | `ValSplitMode`, `TestSplitMode` | `ValSplitMode`, `TestSplitMode` |
| 子类发现 | `__subclasses__()` 自动注册 | `@register_model` 显式注册 |
| 双重继承 | 所有组件均 `nn.Module + Callback` | 所有组件均 `nn.Module + Callback` |
| 三态解析 | `_resolve_component(bool\|Instance\|None)` | `_resolve_component(bool\|Instance\|None)` |
| 配置实例化 | `from_config()` + jsonargparse | `from_config()` + jsonargparse |

## 领域差异

| 维度 | anomalib | torchsense |
|------|----------|------------|
| 模态 | 图像、视频、深度 | IMU/SCG、雷达、音频、多模态 |
| 数据格式 | 图像文件 + DataFrame | HDF5 + 时序信号 |
| 预处理 | Resize + ImageNet Normalize（PreProcessor） | 带通滤波 + 归一化 + 分段（DataModule） + 增强（PreProcessor） |
| 后处理 | 异常分数归一化/阈值 | 信号 clamp / BP 范围校准 |
| 指标 | AUROC, AUPR, F1（异常检测） | SNR, PCC, QRS 检测（信号质量） |
| Backbone | WideResNet, FeatureExtractor, DinoV2 | UNet1d, ResNet1d |
| 导出 | torch + onnx + openvino | ONNX |
| 模型数量 | 22+ 异常检测模型 | 2 模型（SCG2ECG, SCG2BP） |

## 扩展指南

添加新模态只需：

```
models/
├── imu/          # 已有
│   ├── scg2ecg/
│   └── scg2bp/
├── radar/        # 新增
│   └── gesture/
│       ├── __init__.py
│       ├── torch_model.py
│       └── lightning_model.py
└── audio/        # 新增
    └── enhancement/
        ├── __init__.py
        ├── torch_model.py
        └── lightning_model.py
```

每个新模型只需：

1. 继承 `TorchSenseModule`
2. 实现 `model_step()` 和 `_compute_loss()`
3. 声明 `task_type` 和 `modality` 属性
4. 用 `@register_model("name")` 注册
5. 可选：覆盖 `configure_post_processor()` / `configure_evaluator()` / `configure_visualizer()`

添加新数据集只需：

1. 继承 `TorchSenseDataset`，实现 `__getitem__`、`__len__`、`subset()`
2. 暴露 `subject_ids` 属性（如需 subject-based 划分）
3. 继承 `TorchSenseDataModule`，实现 `_setup()` 加载数据到 `self._full_dataset`
4. 划分策略通过 `test_split_mode` / `val_split_mode` 参数配置，无需写划分代码
