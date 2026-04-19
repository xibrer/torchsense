"""Smoke test: verify all key modules can be imported and basic forward passes work."""

import torch
from torchsense.models.imu.scg2ecg import SCG2ECG
from torchsense.models.imu.scg2bp import SCG2BP
from torchsense.models import list_models


def main():
    print("=" * 60)
    print("TorchSense Smoke Tests")
    print("=" * 60)
    print()
    print("Registered models:", list_models())
    print()

    # ==============================================
    # 1. SCG2ECG model
    # ==============================================
    model = SCG2ECG(in_channels=1, out_channels=1, base_filters=32, depth=4)
    print(f"SCG2ECG name={model.name}, task={model.task_type.value}, modality={model.modality.value}")

    x = torch.randn(2, 1, 2048)
    y = model(x)
    print(f"  forward(tensor): {x.shape} -> {y.shape}")

    batch = {"input": x, "target": torch.randn(2, 1, 2048)}
    output = model.model_step(batch)
    print(f"  model_step: prediction={output['prediction'].shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    callbacks = model.configure_callbacks()
    cb_names = [type(c).__name__ for c in callbacks]
    print(f"  Callbacks: {cb_names}")
    assert "PreProcessor" in cb_names
    assert "SignalPostProcessor" in cb_names
    assert "Evaluator" in cb_names
    print()

    # ==============================================
    # 2. SCG2BP model (clean model_step, no pre-split)
    # ==============================================
    bp_model = SCG2BP(in_channels=1)
    print(f"SCG2BP name={bp_model.name}, task={bp_model.task_type.value}, modality={bp_model.modality.value}")

    bp_x = torch.randn(2, 1, 2048)
    bp_y = bp_model(bp_x)
    print(f"  forward(tensor): {bp_x.shape} -> {bp_y.shape}")

    bp_batch = {"input": bp_x, "target": torch.randn(2, 2)}
    bp_output = bp_model.model_step(bp_batch)
    print(f"  model_step: prediction={bp_output['prediction'].shape}")
    assert "sbp_pred" not in bp_output, "model_step should NOT pre-split BP columns"

    bp_params = sum(p.numel() for p in bp_model.parameters())
    print(f"  Parameters: {bp_params:,}")

    bp_cbs = [type(c).__name__ for c in bp_model.configure_callbacks()]
    print(f"  Callbacks: {bp_cbs}")
    assert "BPPostProcessor" in bp_cbs
    assert "Evaluator" in bp_cbs
    print()

    # ==============================================
    # 3. Three-state component resolution
    # ==============================================
    print("Component resolution:")

    m1 = SCG2ECG(evaluator=True)
    assert m1.evaluator is not None
    print("  evaluator=True  -> created")

    m2 = SCG2ECG(evaluator=False)
    assert m2.evaluator is None
    print("  evaluator=False -> None")

    m3 = SCG2ECG(evaluator=None)
    assert m3.evaluator is None
    print("  evaluator=None  -> None")

    from torchsense.metrics.evaluator import Evaluator
    custom_eval = Evaluator()
    m4 = SCG2ECG(evaluator=custom_eval)
    assert m4.evaluator is custom_eval
    print("  evaluator=instance -> used as-is")
    print()

    # ==============================================
    # 4. PreProcessor standalone
    # ==============================================
    from torchsense.pre_processing import PreProcessor
    from torchsense.data.transforms.imu import AddGaussianNoise

    pre = PreProcessor(transform=AddGaussianNoise(std=0.05))
    signal = torch.randn(1, 2048)
    processed = pre(signal)
    assert processed.shape == signal.shape
    print(f"PreProcessor: {signal.shape} -> {processed.shape}")

    # ==============================================
    # 5. PostProcessor standalone
    # ==============================================
    from torchsense.post_processing import SignalPostProcessor, BPPostProcessor

    sig_post = SignalPostProcessor(clamp_min=-3.0, clamp_max=3.0)
    ecg_pred = torch.randn(2, 1, 2048) * 5
    clamped = sig_post(ecg_pred)
    assert clamped.min() >= -3.0 and clamped.max() <= 3.0
    print(f"SignalPostProcessor: clamped range [{clamped.min():.1f}, {clamped.max():.1f}]")

    bp_post = BPPostProcessor(sbp_range=(70, 200), dbp_range=(40, 120))
    bp_raw = torch.tensor([[300.0, 10.0], [90.0, 80.0]])
    bp_clamped = bp_post(bp_raw)
    assert bp_clamped[0, 0] == 200.0 and bp_clamped[0, 1] == 40.0
    print(f"BPPostProcessor: {bp_raw.tolist()} -> {bp_clamped.tolist()}")
    print()

    # ==============================================
    # 6. TorchSenseMetric: fields-based (simple)
    # ==============================================
    from torchsense.metrics.base import TorchSenseMetric
    from torchmetrics import MeanSquaredError

    tsm = TorchSenseMetric(MeanSquaredError(), fields=["prediction", "target"], flatten=True)
    step_out = {
        "prediction": torch.randn(4, 1, 128),
        "target": torch.randn(4, 1, 128),
        "input": torch.randn(4, 1, 128),
    }
    tsm.update(step_out)
    print(f"TorchSenseMetric (fields): name={tsm.name}, value={tsm.compute():.4f}")

    # ==============================================
    # 7. TorchSenseMetric: strict mode
    # ==============================================
    strict_m = TorchSenseMetric(MeanSquaredError(), fields=["prediction", "nonexistent"], strict=True)
    try:
        strict_m.update(step_out)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Strict mode: caught ValueError -> {e}")

    nonstrict_m = TorchSenseMetric(MeanSquaredError(), fields=["prediction", "nonexistent"], strict=False)
    nonstrict_m.update(step_out)
    assert not nonstrict_m.update_called
    print("Non-strict mode: silently skipped")

    # ==============================================
    # 8. TorchSenseMetric: subclass extract() for BP
    # ==============================================
    from torchsense.models.imu.scg2bp.lightning_model import _BPColumnMetric
    from torchmetrics import MeanAbsoluteError

    mae_sbp = _BPColumnMetric(MeanAbsoluteError(), column=0, prefix="sbp_")
    mae_dbp = _BPColumnMetric(MeanAbsoluteError(), column=1, prefix="dbp_")

    bp_step_out = {
        "prediction": torch.tensor([[120.0, 80.0], [130.0, 85.0]]),
        "target": torch.tensor([[118.0, 78.0], [135.0, 90.0]]),
    }
    mae_sbp.update(bp_step_out)
    mae_dbp.update(bp_step_out)
    print(f"_BPColumnMetric SBP: name={mae_sbp.name}, MAE={mae_sbp.compute():.1f}")
    print(f"_BPColumnMetric DBP: name={mae_dbp.name}, MAE={mae_dbp.compute():.1f}")
    print()

    # ==============================================
    # 9. Metrics
    # ==============================================
    from torchsense.metrics.signal import SignalToNoiseRatio, PearsonCorrelation

    snr = SignalToNoiseRatio()
    pcc = PearsonCorrelation()
    p = torch.randn(4, 1, 2048)
    t = p + torch.randn_like(p) * 0.1
    snr.update(p, t)
    pcc.update(p, t)
    print(f"SNR: {snr.compute():.2f} dB")
    print(f"PCC: {pcc.compute():.4f}")
    print()

    # ==============================================
    # 10. ONNX export
    # ==============================================
    import tempfile, os

    export_model = SCG2ECG(in_channels=1, out_channels=1, base_filters=16, depth=2,
                           evaluator=False, visualizer=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = export_model.export_onnx(
            os.path.join(tmpdir, "scg2ecg.onnx"),
            input_shape=(1, 1, 2048),
        )
        assert onnx_path.exists()
        size_kb = onnx_path.stat().st_size / 1024
        print(f"ONNX export: {onnx_path.name} ({size_kb:.0f} KB)")
    print()

    # ==============================================
    # 11. Engine
    # ==============================================
    from torchsense.engine import Engine

    engine = Engine(max_epochs=1, default_root_dir="./test_results")
    print(f"Engine: max_epochs={engine.max_epochs}, root={engine.default_root_dir}")
    print()

    print("=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
