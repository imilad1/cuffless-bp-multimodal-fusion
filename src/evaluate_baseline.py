import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_ecg_baseline.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
VAL_CACHE_PATH = RESULTS_DIR / "cache" / "val_split.npz"
METRICS_CSV_PATH = RESULTS_DIR / "baseline_evaluation_metrics.csv"
METRICS_JSON_PATH = RESULTS_DIR / "baseline_evaluation_metrics.json"
ACTUAL_VS_PRED_PATH = PLOTS_DIR / "baseline_actual_vs_predicted.png"
BLAND_ALTMAN_PATH = PLOTS_DIR / "baseline_bland_altman.png"
EXPECTED_VALIDATION_SAMPLES = 132

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import ChannelMetrics, calculate_metrics, extract_model_state_dict, normalise_ecg_like_training, print_key_value, print_section, select_device
from models.ecg_branch import ECGBaselineModel


class BaselineEvaluationError(RuntimeError):
    pass


def ensure_output_directories() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cached_validation_data() -> tuple[np.ndarray, np.ndarray]:
    if not VAL_CACHE_PATH.exists():
        raise FileNotFoundError(f"Validation cache not found: {VAL_CACHE_PATH}")

    with np.load(VAL_CACHE_PATH, allow_pickle=False) as cached:
        required_keys = {"X_ecg_val", "y_val"}
        missing_keys = required_keys.difference(cached.files)
        if missing_keys:
            raise BaselineEvaluationError(f"Validation cache missing keys: {sorted(missing_keys)}")
        X_ecg_val = cached["X_ecg_val"]
        y_val = cached["y_val"]

    if X_ecg_val.ndim != 2:
        raise BaselineEvaluationError(f"Expected X_ecg_val shape (N, seq_len), got {X_ecg_val.shape}")
    if y_val.ndim != 2 or y_val.shape[1] != 2:
        raise BaselineEvaluationError(f"Expected y_val shape (N, 2), got {y_val.shape}")
    if X_ecg_val.shape[0] != y_val.shape[0]:
        raise BaselineEvaluationError(f"Validation sample count mismatch: ECG={X_ecg_val.shape[0]}, y={y_val.shape[0]}")
    if y_val.shape[0] != EXPECTED_VALIDATION_SAMPLES:
        raise BaselineEvaluationError(f"Expected exactly {EXPECTED_VALIDATION_SAMPLES} validation samples, got {y_val.shape[0]}")

    return X_ecg_val, y_val


def to_tensors(X_ecg_val: np.ndarray, y_val: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    X_ecg_norm = normalise_ecg_like_training(X_ecg_val)
    X_ecg_tensor = torch.from_numpy(X_ecg_norm).unsqueeze(1)
    y_tensor = torch.from_numpy(y_val.astype(np.float32))
    return X_ecg_tensor, y_tensor


def load_baseline_model(device: torch.device) -> torch.nn.Module:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {CHECKPOINT_PATH}")

    model = ECGBaselineModel(dropout=0.2)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_baseline_inference(model: torch.nn.Module, X_ecg_tensor: torch.Tensor, device: torch.device, batch_size: int = 64) -> np.ndarray:
    dataset = TensorDataset(X_ecg_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prediction_batches = []

    with torch.no_grad():
        for (batch_ecg,) in loader:
            batch_ecg = batch_ecg.to(device)
            batch_predictions = model(batch_ecg)
            if batch_predictions.ndim != 2 or batch_predictions.shape[1] != 2:
                raise BaselineEvaluationError(f"Expected model output shape (B, 2), got {tuple(batch_predictions.shape)}")
            prediction_batches.append(batch_predictions.detach().cpu().numpy())

    if not prediction_batches:
        raise BaselineEvaluationError("No prediction batches were produced.")

    predictions = np.vstack(prediction_batches)
    if predictions.shape[0] != EXPECTED_VALIDATION_SAMPLES or predictions.shape[1] != 2:
        raise BaselineEvaluationError(f"Expected predictions shape ({EXPECTED_VALIDATION_SAMPLES}, 2), got {predictions.shape}")
    return predictions


def save_metrics_outputs(metrics: Dict[str, ChannelMetrics], n_samples: int, device: torch.device) -> None:
    rows = [asdict(metrics["Systolic"]), asdict(metrics["Diastolic"])]
    pd.DataFrame(rows).to_csv(METRICS_CSV_PATH, index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "validation_samples": int(n_samples),
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_import_path": "models.ecg_branch.ECGBaselineModel",
        "validation_data_source": "cache",
        "validation_cache_path": str(VAL_CACHE_PATH),
        "metrics": rows,
        "units": {
            "mae": "mmHg",
            "rmse": "mmHg",
            "sde": "mmHg",
            "mean_bias": "mmHg",
            "loa_upper": "mmHg",
            "loa_lower": "mmHg",
            "pearson_r": "unitless",
        },
    }
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def add_metric_box(ax: plt.Axes, metrics: ChannelMetrics) -> None:
    text = (
        f"MAE = {metrics.mae:.2f} mmHg\n"
        f"RMSE = {metrics.rmse:.2f} mmHg\n"
        f"r = {metrics.pearson_r:.2f}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.5", "alpha": 0.88},
    )


def plot_actual_vs_predicted(y_true: np.ndarray, predictions: np.ndarray, metrics: Dict[str, ChannelMetrics]) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_specs = [
        (axes[0], "Systolic", y_true[:, 0], predictions[:, 0], "crimson"),
        (axes[1], "Diastolic", y_true[:, 1], predictions[:, 1], "navy"),
    ]

    for ax, channel, true_values, predicted_values, color in plot_specs:
        lo = float(min(true_values.min(), predicted_values.min()) - 5.0)
        hi = float(max(true_values.max(), predicted_values.max()) + 5.0)
        ax.scatter(true_values, predicted_values, color=color, alpha=0.6, s=42, edgecolors="none", label="Validation samples")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.6, label="Perfect accuracy")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{channel} Baseline Predicted vs Actual", fontweight="bold")
        ax.set_xlabel("True Blood Pressure (mmHg)")
        ax.set_ylabel("Predicted Blood Pressure (mmHg)")
        add_metric_box(ax, metrics[channel])
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle(f"ECG Baseline Model Validation Set (N={len(y_true)})", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ACTUAL_VS_PRED_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_horizontal_line(ax: plt.Axes, y_value: float, label: str, x_position: float, color: str, va: str) -> None:
    ax.text(
        x_position,
        y_value,
        label,
        color=color,
        fontsize=9,
        va=va,
        ha="right",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def plot_bland_altman_channel(ax: plt.Axes, channel: str, true_values: np.ndarray, predicted_values: np.ndarray, color: str, metrics: ChannelMetrics) -> None:
    averages = (true_values + predicted_values) / 2.0
    errors = predicted_values - true_values
    x_min = float(averages.min() - 5.0)
    x_max = float(averages.max() + 5.0)

    ax.fill_between([x_min, x_max], metrics.loa_lower, metrics.loa_upper, color="lightsteelblue", alpha=0.25, label="95% LoA band")
    ax.scatter(averages, errors, color=color, alpha=0.6, s=42, edgecolors="none", label="Predictions")
    ax.axhline(metrics.mean_bias, color="black", linestyle="-", linewidth=1.8, label="Mean Bias")
    ax.axhline(metrics.loa_upper, color="dimgray", linestyle="--", linewidth=1.6, label="± 1.96 SDE (95% LoA)")
    ax.axhline(metrics.loa_lower, color="dimgray", linestyle="--", linewidth=1.6)
    annotate_horizontal_line(ax, metrics.mean_bias, f"Mean Bias: {metrics.mean_bias:+.2f}", x_max, "black", "bottom")
    annotate_horizontal_line(ax, metrics.loa_upper, f"Upper LoA: {metrics.loa_upper:+.2f}", x_max, "dimgray", "bottom")
    annotate_horizontal_line(ax, metrics.loa_lower, f"Lower LoA: {metrics.loa_lower:+.2f}", x_max, "dimgray", "top")
    ax.set_xlim(x_min, x_max)
    y_margin = max(5.0, float(np.std(errors) * 0.5))
    ax.set_ylim(float(min(errors.min(), metrics.loa_lower) - y_margin), float(max(errors.max(), metrics.loa_upper) + y_margin))
    ax.set_title(f"{channel} Baseline Bland-Altman", fontweight="bold")
    ax.set_xlabel("Mean of True and Predicted (mmHg)")
    ax.set_ylabel("Prediction Error: Pred - True (mmHg)")
    ax.legend(loc="best", frameon=True)


def plot_bland_altman(y_true: np.ndarray, predictions: np.ndarray, metrics: Dict[str, ChannelMetrics]) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_bland_altman_channel(axes[0], "Systolic", y_true[:, 0], predictions[:, 0], "crimson", metrics["Systolic"])
    plot_bland_altman_channel(axes[1], "Diastolic", y_true[:, 1], predictions[:, 1], "navy", metrics["Diastolic"])
    fig.suptitle("ECG Baseline Model Bland-Altman Agreement", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(BLAND_ALTMAN_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def assert_metrics_are_valid(metrics: Dict[str, ChannelMetrics]) -> None:
    for channel, channel_metrics in metrics.items():
        for field_name in ("mae", "rmse"):
            value = getattr(channel_metrics, field_name)
            if math.isnan(value) or value == 0.0:
                raise BaselineEvaluationError(f"Invalid {channel} {field_name}: {value}")


def run_evaluation() -> None:
    ensure_output_directories()
    device = select_device()
    print_section("BASELINE EVALUATION INITIALISATION")
    print_key_value("Project root", PROJECT_ROOT)
    print_key_value("Selected device", device)
    print_key_value("Checkpoint path", CHECKPOINT_PATH)
    print_key_value("Validation cache", VAL_CACHE_PATH)

    X_ecg_val, y_val = load_cached_validation_data()
    X_ecg_tensor, y_tensor = to_tensors(X_ecg_val, y_val)
    model = load_baseline_model(device)
    predictions = run_baseline_inference(model, X_ecg_tensor, device)
    y_true = y_tensor.detach().cpu().numpy()

    metrics = {
        "Systolic": calculate_metrics("Systolic", y_true[:, 0], predictions[:, 0]),
        "Diastolic": calculate_metrics("Diastolic", y_true[:, 1], predictions[:, 1]),
    }
    assert_metrics_are_valid(metrics)
    save_metrics_outputs(metrics, y_true.shape[0], device)
    plot_actual_vs_predicted(y_true, predictions, metrics)
    plot_bland_altman(y_true, predictions, metrics)

    print_section("BASELINE PRIMARY ERROR METRICS")
    print("+-----------+----------+----------+----------+----------+----------+")
    print("| Channel   | MAE      | RMSE     | SDE      | Bias     | r        |")
    print("+-----------+----------+----------+----------+----------+----------+")
    for key in ("Systolic", "Diastolic"):
        m = metrics[key]
        print(f"| {m.channel:<9} |{m.mae:>8.2f} |{m.rmse:>8.2f} |{m.sde:>8.2f} |{m.mean_bias:>8.2f} |{m.pearson_r:>8.2f} |")
    print("+-----------+----------+----------+----------+----------+----------+")

    print_section("BASELINE OUTPUT ARTEFACTS")
    print_key_value("Validation samples", y_true.shape[0])
    print_key_value("Metrics CSV", METRICS_CSV_PATH)
    print_key_value("Metrics JSON", METRICS_JSON_PATH)
    print_key_value("Predicted vs Actual", ACTUAL_VS_PRED_PATH)
    print_key_value("Bland-Altman", BLAND_ALTMAN_PATH)
    print("\nBaseline evaluation completed successfully.")


def main() -> int:
    try:
        run_evaluation()
        return 0
    except Exception as exc:
        print("\n" + "!" * 90)
        print("BASELINE EVALUATION FAILED")
        print("!" * 90)
        print(f"{type(exc).__name__}: {exc}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
