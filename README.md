# TorchSense

A modular deep learning framework for intelligent sensing, inspired by [anomalib](https://github.com/openvinotoolkit/anomalib).

Built on **PyTorch Lightning**, TorchSense provides a plug-and-play architecture for physiological signal processing and beyond. The framework follows anomalib's component-based design -- every model assembles four pluggable sub-components (`pre_processor`, `post_processor`, `evaluator`, `visualizer`), all implemented as `nn.Module + Callback` for seamless integration with Lightning's training loop and standalone inference/export.

## Supported Directions

| Modality | Tasks | Status |
|----------|-------|--------|
| IMU / Accelerometer | SCG-to-ECG reconstruction, blood pressure prediction | In Progress |
| mmWave Radar | Point cloud gesture/activity recognition | Planned |
| Audio + IMU | Multi-modal speech enhancement | Planned |

## Installation

```bash
pip install -e .

# with dev dependencies (includes ONNX export)
pip install -e ".[dev]"

# ONNX export only
pip install -e ".[export]"
```

Requires Python >= 3.11.

## Quick Start

### CLI

```bash
# Train SCG-to-ECG reconstruction model
torchsense fit --config configs/model/scg2ecg_unet.yaml --config configs/data/vib2ecg.yaml

# Train SCG-to-BP prediction model
torchsense fit --config configs/model/scg2bp_resnet.yaml --config configs/data/vib2ecg.yaml

# Evaluate
torchsense test --config configs/model/scg2ecg_unet.yaml --config configs/data/vib2ecg.yaml
```

### Python API

```python
from torchsense.engine import Engine
from torchsense.models.imu.scg2ecg import SCG2ECG
from torchsense.data.datamodules.imu.vib2ecg import Vib2ECGDataModule

model = SCG2ECG(in_channels=1, out_channels=1)
datamodule = Vib2ECGDataModule(data_root="data/vib2ecg", batch_size=32)
engine = Engine(max_epochs=50)

# Train + Test
engine.train(model=model, datamodule=datamodule)

# Export to ONNX
engine.export(model, output_path="scg2ecg.onnx")
```

### Component Configuration

Every model supports pluggable components with three-state configuration:

```python
model = SCG2ECG(
    evaluator=True,          # use default Evaluator (MSE + PCC)
    visualizer=False,        # disable visualization
    post_processor=custom,   # pass a custom instance
)
```

## Architecture

TorchSense follows anomalib's modular design built on PyTorch Lightning. Each model is a `TorchSenseModule` that orchestrates four sub-components:

```
                    +-------------------------------------------+
                    |          TorchSenseModule                  |
                    |  (ExportMixin + LightningModule)           |
                    |                                           |
                    |  +--------------+  +--------------------+ |
                    |  |PreProcessor  |  | PostProcessor      | |
                    |  |(Module+CB)   |  | (Module+CB)        | |
                    |  +------+-------+  +------^-------------+ |
                    |         |                 |               |
                    |         v                 |               |
                    |  +------------------------+--+            |
                    |  |   model (nn.Module)        |            |
                    |  +---------------------------+            |
                    |                                           |
                    |  +--------------+  +--------------------+ |
                    |  | Evaluator    |  |  Visualizer        | |
                    |  | (Module+CB)  |  |  (Module+CB)       | |
                    |  +--------------+  +--------------------+ |
                    +-------------------------------------------+
```

- **Engine** -- Orchestrates training / validation / testing / export via `fit`, `train`, `test`, `export`, `from_config`
- **Models** -- Organized by modality (`imu/`, `radar/`, `audio/`), with shared backbones (UNet1d, ResNet1d) and layers (ConvBlock1d, SqueezeExcitation1d)
- **Data** -- Two-layer design: `TorchSenseDataset` (sample access + `subset()`) and `TorchSenseDataModule` (lifecycle + split strategies via `ValSplitMode` / `TestSplitMode`)
- **Metrics** -- `TorchSenseMetric` wrapper (fields + strict + subclass `extract()`) managed by `Evaluator` (nn.Module + Callback)
- **Pre-Processing** -- `PreProcessor` (nn.Module + Callback) for online tensor transforms (augmentation). Heavy signal preprocessing (bandpass filter, segmentation) lives in DataModule at the numpy level.
- **Post-Processing** -- `PostProcessor` (nn.Module + Callback) for output modification (signal clamping, BP calibration). Participates in ONNX export.
- **Visualization** -- `Visualizer` (nn.Module + Callback): `SignalVisualizer` for waveform comparison, `BPVisualizer` for scatter plots
- **Export** -- `ExportMixin` providing ONNX export via `model.export_onnx()`
- **Callbacks** -- `TimerCallback` for training time tracking

## Project Structure

```
src/torchsense/
├── cli/                 # LightningCLI entry point (torchsense fit/test/validate/predict)
├── engine/              # Engine: fit, validate, test, predict, train, export, from_config
├── models/
│   ├── components/
│   │   ├── base/        # TorchSenseModule, ExportMixin
│   │   ├── backbones/   # UNet1d, ResNet1d
│   │   └── layers/      # ConvBlock1d, SqueezeExcitation1d
│   └── imu/
│       ├── scg2ecg/     # SCG→ECG reconstruction (UNet1d)
│       └── scg2bp/      # SCG→BP prediction (ResNet1d)
├── data/
│   ├── datasets/        # TorchSenseDataset base + Vib2ECGDataset
│   ├── datamodules/     # TorchSenseDataModule base (ValSplitMode, TestSplitMode) + Vib2ECGDataModule
│   └── transforms/      # AddGaussianNoise, RandomScale, RandomTimeShift, Compose
├── metrics/
│   ├── base.py          # TorchSenseMetric (fields + strict + extract subclassing)
│   ├── evaluator.py     # Evaluator (nn.Module + Callback)
│   ├── signal.py        # SignalToNoiseRatio, PearsonCorrelation
│   └── cardiac.py       # QRSDetectionAccuracy
├── pre_processing/
│   ├── pre_processor.py # PreProcessor (nn.Module + Callback)
│   └── imu.py           # bandpass_filter, normalize_signal, segment_signal
├── post_processing/
│   ├── post_processor.py # PostProcessor, SignalPostProcessor, BPPostProcessor
│   └── imu.py           # smooth_ecg, detect_r_peaks, compute_heart_rate
├── visualization/
│   ├── base.py          # Visualizer (nn.Module + Callback)
│   └── signal.py        # SignalVisualizer, BPVisualizer, plot functions
├── callbacks/           # TimerCallback
└── utils/               # Logging, path utilities

configs/
├── data/
│   └── vib2ecg.yaml
└── model/
    ├── scg2ecg_unet.yaml
    └── scg2bp_resnet.yaml

examples/
└── imu/
    └── train_scg2ecg.py
```

## Extending to New Modalities

Adding a new modality requires only:

1. Subclass `TorchSenseModule`
2. Implement `model_step()` and `_compute_loss()`
3. Declare `task_type` and `modality` properties
4. Register with `@register_model("name")`
5. Optionally override `configure_evaluator()` / `configure_post_processor()` / `configure_visualizer()`

```python
@register_model("radar_gesture")
class RadarGesture(TorchSenseModule):
    task_type = TaskType.CLASSIFICATION
    modality = ModalityType.RADAR

    def model_step(self, batch):
        logits = self.model(batch["input"])
        return {"prediction": logits}

    def _compute_loss(self, output, batch):
        return F.cross_entropy(output["prediction"], batch["target"])
```

Adding a new dataset:

1. Subclass `TorchSenseDataset` -- implement `__getitem__`, `__len__`, `subset()`
2. Expose `subject_ids` property for subject-based splitting
3. Subclass `TorchSenseDataModule` -- implement `_setup()` to load data into `self._full_dataset`
4. Configure split strategy via `test_split_mode` / `val_split_mode` parameters

## Datasets

### IMU / Accelerometer -- SCG-to-ECG Reconstruction

| Dataset | Description | Subjects | Signals | Source |
|---------|-------------|----------|---------|--------|
| **Vib2ECG** | First paired multi-location chest-lead SCG-ECG dataset. 12-lead ECG + 6 chest-lead IMU vibration signals at 1kHz. Includes U-Net benchmark. | 17 | ECG (12-lead) + IMU/SCG (6-axis x 6 locations) | [IEEE DataPort](https://ieee-dataport.org/documents/vib2ecg-paired-chest-lead-scg-ecg-dataset-and-benchmark-ecg-reconstruction) / [arXiv:2603.15539](https://arxiv.org/abs/2603.15539) |
| **CEBS** | Combined ECG, Breathing and Seismocardiograms. 2-lead ECG + triaxial SCG at 5kHz, three phases (baseline/music/post). | 20 | ECG (2-lead) + SCG (3-axis) + Respiration | [PhysioNet](https://physionet.org/content/cebsdb/) |
| **SensSmartTech** | Synchronous ECG, PCG, PPG and accelerometer (SCG) at wide heart rate range (52-182 BPM) via treadmill. | -- | ECG + PCG + PPG + SCG | [PhysioNet](https://physionet.org/content/senssmarttech/1.0.0/) |

### IMU / Accelerometer -- SCG-to-Blood Pressure Prediction

| Dataset | Description | Subjects | Signals | Source |
|---------|-------------|----------|---------|--------|
| **SCG-RHC** | First public wearable SCG + right heart catheter dataset. ECG (1kHz) + triaxial SCG (500Hz) with invasive PAP, PCWP, arterial BP ground truth. Includes demographics and hemodynamic values. | 73 | ECG + SCG (3-axis) + RHC Pressure + Arterial BP | [PhysioNet](https://physionet.org/content/scg-rhc-wearable-database/1.0.0/) |
| **Valvular Heart Disease DB** | ECG, SCG and GCG from patients with valvular heart diseases. Includes echo parameters (ejection fraction, valve area, mean gradient pressure). | 100 | ECG + SCG + GCG + Echo parameters | [Frontiers](https://doi.org/10.3389/fphys.2021.750221) |

### IMU / Accelerometer -- Supplementary

| Dataset | Description | Subjects | Source |
|---------|-------------|----------|--------|
| **MSCardio SCG** | Large-scale smartphone-based SCG recordings (iOS/Android) with demographics. SCG-only, useful for pretraining or signal processing research. | 108 (502 recordings) | [Zenodo](https://zenodo.org/records/14975878) |
| **SCG Signal Processing** | 1000 samples with vertical-axis accelerometer data, noise levels, and heart rate. | -- | [IEEE DataPort](https://ieee-dataport.org/documents/seismocardiography-scg-signal-processing-dataset) |

### mmWave Radar -- Human Pose / Joint Estimation

| Dataset | Scale | Subjects | Modalities | Annotations | Source |
|---------|-------|----------|------------|-------------|--------|
| **MMVR** | 345K frames / 7.59M keypoints | 25 | mmWave multi-view | 2D keypoints + bbox + segmentation mask | [Zenodo](https://zenodo.org/records/12611978) / [arXiv:2406.10708](https://arxiv.org/abs/2406.10708) |
| **mRI** | 160K+ frames | 20 | mmWave + RGB-D + IMU | 3D joint positions + action labels | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffpp) / [GitHub](https://github.com/SizheAn/mRI) |
| **mmBody** | Multi-scene | -- | mmWave + RGB(D) | 3D skeleton + mesh | [GitHub](https://github.com/Chen3110/mmBody) (email request) |
| **M4Human** | 661K frames | 20 | mmWave + RGB + Depth + MoCap | 3D mesh + global trajectory | [arXiv:2512.12378](https://arxiv.org/html/2512.12378v1) (pending release) |
| **HuPR** | 141K frames | 6 | mmWave + RGB | 2D keypoints | [WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Lee_HuPR_A_Benchmark_for_Human_Pose_Estimation_Using_Millimeter_Wave_WACV_2023_paper.html) |
| **mmWave Pose** | 7,397 frames | -- | mmWave point cloud | 19 joint labels (Kinect GT) | [IEEE DataPort](https://ieee-dataport.org/documents/mmwave-pose-estimation) |

### Audio + Vibration -- Multi-modal Speech Enhancement

| Dataset | Duration | Speakers | Sensors | Language | Source |
|---------|----------|----------|---------|----------|--------|
| **Vibravox** | 45.5h | 188 | Bone conduction x2 + in-ear mic x2 + laryngophone + air mic (reference) | French | [HuggingFace](https://huggingface.co/datasets/Cnam-LMSSC/vibravox) / [arXiv:2407.11828](https://arxiv.org/html/2407.11828v2) |
| **TAPS** | 6,000 utterances | 60 | Throat mic + acoustic mic (paired) | Korean | [HuggingFace](https://huggingface.co/datasets/yskim3271/Throat_and_Acoustic_Pairing_Speech_Dataset) / [arXiv:2502.11478](https://arxiv.org/html/2502.11478v2) |

Reference methods (data not public):

| Method | Approach | Key Result | Paper |
|--------|----------|------------|-------|
| **SEANet** (Google) | Earbud accelerometer + mic, wave-to-wave ConvNet | Robust to same-loudness interfering speech | [arXiv:2009.02095](https://arxiv.org/abs/2009.02095) / [GitHub](https://github.com/google-research/seanet) |
| **VibOmni** | IMU + mic on earables, BCF data augmentation | PESQ +21%, SNR +26%, WER -40% | [arXiv:2512.02515](https://arxiv.org/abs/2512.02515) |
| **BCDM** | Bone-conduction guided conditional diffusion | Uses Vibravox dataset | [GitHub](https://github.com/sp-uhh/bcdm) |

## License

Apache-2.0
