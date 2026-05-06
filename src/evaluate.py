"""
Final clinical evaluation and visualization script for the ECG + CGM fusion model.
"""

import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "best_fusion_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CACHE_DIR = RESULTS_DIR / "cache"
VAL_CACHE_PATH = CACHE_DIR / "val_split.npz"
METRICS_CSV_PATH = RESULTS_DIR / "evaluation_metrics.csv"
METRICS_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"
ACTUAL_VS_PRED_PATH = PLOTS_DIR / "actual_vs_predicted.png"
BLAND_ALTMAN_PATH = PLOTS_DIR / "bland_altman.png"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ChannelMetrics:
    channel: str
    mae: float
    rmse: float
    sde: float
    pearson_r: float
    mean_bias: float
    loa_upper: float
    loa_lower: float


class EvaluationError(RuntimeError):
    pass


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_output_directories() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    width = 90
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)


def print_key_value(label: str, value: object) -> None:
    print(f"  {label:<28}: {value}")


def checkpoint_is_available() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Trained fusion checkpoint not found. Expected file:\n"
            f"  {CHECKPOINT_PATH}\n"
            "Run training first so that models/checkpoints/best_fusion_model.pth exists."
        )


def checkpoint_metadata() -> Dict[str, object]:
    checkpoint_is_available()
    checkpoint_stat = CHECKPOINT_PATH.stat()
    return {
        "checkpoint_path": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
    }


def cache_status() -> Tuple[bool, str]:
    if not VAL_CACHE_PATH.exists():
        return False, "cache file is missing"
    if not CHECKPOINT_PATH.exists():
        return False, "checkpoint file is missing"

    try:
        with np.load(VAL_CACHE_PATH, allow_pickle=False) as cached:
            required_keys = {"X_ecg_val", "X_cgm_val", "y_val", "checkpoint_path", "checkpoint_mtime_ns"}
            missing_keys = required_keys.difference(cached.files)
            if missing_keys:
                return False, f"cache metadata is incomplete: missing {sorted(missing_keys)}"

            cached_checkpoint_path = str(cached["checkpoint_path"].item())
            cached_checkpoint_mtime_ns = int(cached["checkpoint_mtime_ns"].item())
    except Exception as exc:
        return False, f"cache metadata could not be read: {exc}"

    current_metadata = checkpoint_metadata()
    if cached_checkpoint_path != current_metadata["checkpoint_path"]:
        return False, "checkpoint path changed"
    if cached_checkpoint_mtime_ns != current_metadata["checkpoint_mtime_ns"]:
        return False, "checkpoint modification time changed"
    return True, "checkpoint metadata matches"


def load_validation_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    checkpoint_is_available()

    valid_cache, cache_reason = cache_status()
    if valid_cache:
        try:
            with np.load(VAL_CACHE_PATH, allow_pickle=False) as cached:
                X_ecg_val = cached["X_ecg_val"]
                X_cgm_val = cached["X_cgm_val"]
                y_val = cached["y_val"]
            validate_array_shapes(X_ecg_val, X_cgm_val, y_val)
            print(f"  [CACHE] Using validation cache ({cache_reason}).")
            return X_ecg_val, X_cgm_val, y_val, "cache"
        except Exception as exc:
            print(f"  [CACHE] Existing validation cache could not be used: {exc}")
            print("  [CACHE] Rebuilding validation split from raw data.")
    else:
        print(f"  [CACHE] Cache miss: {cache_reason}.")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Raw data path not found and validation cache cannot be used. Expected directory:\n"
            f"  {RAW_DATA_PATH}"
        )

    try:
        from train import prepare_global_dataset
    except Exception as exc:
        raise EvaluationError(
            "Could not import prepare_global_dataset from src/train.py."
        ) from exc

    try:
        print("  [DATA] Building validation split via prepare_global_dataset(data/raw).")
        print("  [DATA] This may take several minutes because raw ECG files are re-aligned.")
        _, X_ecg_val, _, X_cgm_val, _, y_val = prepare_global_dataset(RAW_DATA_PATH)
        validate_array_shapes(X_ecg_val, X_cgm_val, y_val)
        metadata = checkpoint_metadata()
        np.savez_compressed(
            VAL_CACHE_PATH,
            X_ecg_val=X_ecg_val,
            X_cgm_val=X_cgm_val,
            y_val=y_val,
            checkpoint_path=metadata["checkpoint_path"],
            checkpoint_mtime_ns=metadata["checkpoint_mtime_ns"],
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        print(f"  [CACHE] Validation split cached at {VAL_CACHE_PATH}.")
        return X_ecg_val, X_cgm_val, y_val, "rebuilt"
    except Exception as exc:
        raise EvaluationError("Failed to prepare validation dataset from raw data.") from exc


def validate_array_shapes(
    X_ecg_val: np.ndarray,
    X_cgm_val: np.ndarray,
    y_val: np.ndarray,
) -> None:
    if X_ecg_val.ndim != 2:
        raise EvaluationError(f"Expected X_ecg_val to have shape (N, seq_len), got {X_ecg_val.shape}")
    if X_cgm_val.ndim != 3 or X_cgm_val.shape[-1] != 2:
        raise EvaluationError(f"Expected X_cgm_val to have shape (N, 7, 2), got {X_cgm_val.shape}")
    if y_val.ndim != 2 or y_val.shape[1] != 2:
        raise EvaluationError(f"Expected y_val to have shape (N, 2), got {y_val.shape}")
    if not (X_ecg_val.shape[0] == X_cgm_val.shape[0] == y_val.shape[0]):
        raise EvaluationError(
            "Validation arrays have inconsistent sample counts: "
            f"ECG={X_ecg_val.shape[0]}, CGM={X_cgm_val.shape[0]}, y={y_val.shape[0]}"
        )
    if y_val.shape[0] == 0:
        raise EvaluationError("Validation split is empty; cannot evaluate model.")


def normalise_ecg_like_training(X_ecg_val: np.ndarray) -> np.ndarray:
    means = X_ecg_val.mean(axis=1, keepdims=True)
    stds = X_ecg_val.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return ((X_ecg_val - means) / stds).astype(np.float32)


def to_tensors(
    X_ecg_val: np.ndarray,
    X_cgm_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_ecg_norm = normalise_ecg_like_training(X_ecg_val)
    X_ecg_tensor = torch.from_numpy(X_ecg_norm).unsqueeze(1)
    X_cgm_tensor = torch.from_numpy(X_cgm_val.astype(np.float32))
    y_tensor = torch.from_numpy(y_val.astype(np.float32))
    return X_ecg_tensor, X_cgm_tensor, y_tensor


def import_model_class():
    try:
        from models.fusion import MultiModalBPRegressor

        return MultiModalBPRegressor, "models.fusion"
    except Exception as exc:
        raise EvaluationError("Could not import MultiModalBPRegressor from models.fusion.") from exc


def extract_model_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise EvaluationError("Checkpoint must be a state_dict or a dictionary containing model weights.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise EvaluationError("Resolved checkpoint weights are not a valid state_dict mapping.")

    if any(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {str(key).replace("module.", "", 1): value for key, value in state_dict.items()}

    return state_dict


def load_model(device: torch.device) -> Tuple[torch.nn.Module, str]:
    checkpoint_is_available()
    ModelClass, import_path = import_model_class()
    model = ModelClass(dropout=0.2)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict = extract_model_state_dict(checkpoint)
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise EvaluationError(f"Failed to load model weights from {CHECKPOINT_PATH}") from exc

    model.to(device)
    model.eval()
    return model, import_path


def run_inference(
    model: torch.nn.Module,
    X_ecg_tensor: torch.Tensor,
    X_cgm_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    try:
        dataset = TensorDataset(X_ecg_tensor, X_cgm_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        prediction_batches = []

        with torch.no_grad():
            for batch_ecg, batch_cgm in loader:
                batch_ecg = batch_ecg.to(device)
                batch_cgm = batch_cgm.to(device)
                batch_predictions = model(batch_ecg, batch_cgm)
                if batch_predictions.ndim != 2 or batch_predictions.shape[1] != 2:
                    raise EvaluationError(f"Expected model output shape (B, 2), got {tuple(batch_predictions.shape)}")
                prediction_batches.append(batch_predictions.detach().cpu().numpy())

        if not prediction_batches:
            raise EvaluationError("No prediction batches were produced.")

        predictions = np.vstack(prediction_batches)
        if predictions.ndim != 2 or predictions.shape[1] != 2:
            raise EvaluationError(f"Expected model output shape (N, 2), got {tuple(predictions.shape)}")
        return predictions
    except Exception as exc:
        raise EvaluationError("Model inference failed.") from exc


def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calculate_metrics(channel: str, y_true: np.ndarray, y_pred: np.ndarray) -> ChannelMetrics:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    sde = float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0
    pearson_r = safe_pearson_r(y_true, y_pred)
    mean_bias = float(np.mean(errors))
    loa_upper = float(mean_bias + 1.96 * sde)
    loa_lower = float(mean_bias - 1.96 * sde)
    return ChannelMetrics(
        channel=channel,
        mae=mae,
        rmse=rmse,
        sde=sde,
        pearson_r=pearson_r,
        mean_bias=mean_bias,
        loa_upper=loa_upper,
        loa_lower=loa_lower,
    )


def format_float(value: float, width: int = 8) -> str:
    if math.isnan(value):
        return f"{'NaN':>{width}}"
    return f"{value:>{width}.2f}"


def print_metrics_report(
    metrics: Dict[str, ChannelMetrics],
    n_samples: int,
    device: torch.device,
    model_import_path: str,
    data_source: str,
    y_true: np.ndarray,
) -> None:
    print_section("FINAL CLINICAL MODEL EVALUATION")
    print_key_value("Validation samples", n_samples)
    print_key_value("Device", device)
    print_key_value("Model import path", model_import_path)
    print_key_value("Checkpoint", CHECKPOINT_PATH)
    print_key_value("Validation data source", data_source)
    print_key_value("Validation cache", VAL_CACHE_PATH)
    print_key_value("Systolic target range", f"{y_true[:, 0].min():.2f} to {y_true[:, 0].max():.2f} mmHg")
    print_key_value("Diastolic target range", f"{y_true[:, 1].min():.2f} to {y_true[:, 1].max():.2f} mmHg")

    print_section("PRIMARY ERROR METRICS")
    print("+-----------+----------+----------+----------+----------+----------+")
    print("| Channel   | MAE      | RMSE     | SDE      | Bias     | r        |")
    print("+-----------+----------+----------+----------+----------+----------+")
    for key in ("Systolic", "Diastolic"):
        m = metrics[key]
        print(
            f"| {m.channel:<9} |"
            f"{format_float(m.mae)} |"
            f"{format_float(m.rmse)} |"
            f"{format_float(m.sde)} |"
            f"{format_float(m.mean_bias)} |"
            f"{format_float(m.pearson_r)} |"
        )
    print("+-----------+----------+----------+----------+----------+----------+")
    print("  Units: MAE, RMSE, SDE, and Bias are reported in mmHg.")
    print("  r: Pearson correlation coefficient between true and predicted BP.")

    print_section("BLAND-ALTMAN AGREEMENT SUMMARY")
    print("+-----------+-------------+-------------+-------------+")
    print("| Channel   | Mean Bias   | Lower LoA   | Upper LoA   |")
    print("+-----------+-------------+-------------+-------------+")
    for key in ("Systolic", "Diastolic"):
        m = metrics[key]
        print(
            f"| {m.channel:<9} |"
            f"{format_float(m.mean_bias, 11)} |"
            f"{format_float(m.loa_lower, 11)} |"
            f"{format_float(m.loa_upper, 11)} |"
        )
    print("+-----------+-------------+-------------+-------------+")
    print("  Bias and limits of agreement are computed from Predicted - True error.")


def save_metrics_outputs(
    metrics: Dict[str, ChannelMetrics],
    n_samples: int,
    device: torch.device,
    model_import_path: str,
    data_source: str,
) -> None:
    rows = [asdict(metrics["Systolic"]), asdict(metrics["Diastolic"])]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "validation_samples": int(n_samples),
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_import_path": model_import_path,
        "validation_data_source": data_source,
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


def plot_actual_vs_predicted(
    y_true_sys: np.ndarray,
    y_true_dia: np.ndarray,
    y_pred_sys: np.ndarray,
    y_pred_dia: np.ndarray,
    metrics: Dict[str, ChannelMetrics],
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_specs = [
        (axes[0], "Systolic", y_true_sys, y_pred_sys, "crimson"),
        (axes[1], "Diastolic", y_true_dia, y_pred_dia, "navy"),
    ]

    for ax, channel, y_true, y_pred, color in plot_specs:
        lo = float(min(y_true.min(), y_pred.min()) - 5.0)
        hi = float(max(y_true.max(), y_pred.max()) + 5.0)
        ax.scatter(y_true, y_pred, color=color, alpha=0.6, s=42, edgecolors="none", label="Validation samples")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.6, label="Perfect accuracy")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{channel}: Predicted vs Actual", fontweight="bold")
        ax.set_xlabel("True Blood Pressure (mmHg)")
        ax.set_ylabel("Predicted Blood Pressure (mmHg)")
        add_metric_box(ax, metrics[channel])
        ax.legend(loc="lower right", frameon=True)

    fig.suptitle(
        f"Multi-Modal ECG + CGM Fusion Model: Validation Set (N={len(y_true_sys)})",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ACTUAL_VS_PRED_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_horizontal_line(
    ax: plt.Axes,
    y_value: float,
    label: str,
    x_position: float,
    color: str,
    va: str,
) -> None:
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


def plot_bland_altman_channel(
    ax: plt.Axes,
    channel: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    color: str,
    metrics: ChannelMetrics,
) -> None:
    averages = (y_true + y_pred) / 2.0
    errors = y_pred - y_true
    x_min = float(averages.min() - 5.0)
    x_max = float(averages.max() + 5.0)

    ax.fill_between(
        [x_min, x_max],
        metrics.loa_lower,
        metrics.loa_upper,
        color="lightsteelblue",
        alpha=0.25,
        label="95% LoA band",
    )
    ax.scatter(averages, errors, color=color, alpha=0.6, s=42, edgecolors="none", label="Predictions")
    ax.axhline(metrics.mean_bias, color="black", linestyle="-", linewidth=1.8, label="Mean Bias")
    ax.axhline(metrics.loa_upper, color="dimgray", linestyle="--", linewidth=1.6, label="± 1.96 SDE (95% LoA)")
    ax.axhline(metrics.loa_lower, color="dimgray", linestyle="--", linewidth=1.6)

    annotate_horizontal_line(
        ax,
        metrics.mean_bias,
        f"Mean Bias: {metrics.mean_bias:+.2f}",
        x_max,
        "black",
        "bottom",
    )
    annotate_horizontal_line(
        ax,
        metrics.loa_upper,
        f"Upper LoA: {metrics.loa_upper:+.2f}",
        x_max,
        "dimgray",
        "bottom",
    )
    annotate_horizontal_line(
        ax,
        metrics.loa_lower,
        f"Lower LoA: {metrics.loa_lower:+.2f}",
        x_max,
        "dimgray",
        "top",
    )

    ax.set_xlim(x_min, x_max)
    y_margin = max(5.0, float(np.std(errors) * 0.5))
    ax.set_ylim(float(min(errors.min(), metrics.loa_lower) - y_margin), float(max(errors.max(), metrics.loa_upper) + y_margin))
    ax.set_title(f"{channel}: Bland-Altman", fontweight="bold")
    ax.set_xlabel("Mean of True and Predicted (mmHg)")
    ax.set_ylabel("Prediction Error: Pred - True (mmHg)")
    ax.legend(loc="best", frameon=True)


def plot_bland_altman(
    y_true_sys: np.ndarray,
    y_true_dia: np.ndarray,
    y_pred_sys: np.ndarray,
    y_pred_dia: np.ndarray,
    metrics: Dict[str, ChannelMetrics],
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_bland_altman_channel(axes[0], "Systolic", y_true_sys, y_pred_sys, "crimson", metrics["Systolic"])
    plot_bland_altman_channel(axes[1], "Diastolic", y_true_dia, y_pred_dia, "navy", metrics["Diastolic"])
    fig.suptitle("Multi-Modal ECG + CGM Fusion Model: Bland-Altman Agreement", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(BLAND_ALTMAN_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_evaluation() -> None:
    ensure_output_directories()
    device = select_device()

    print_section("INITIALISATION")
    print_key_value("Project root", PROJECT_ROOT)
    print_key_value("Raw data path", RAW_DATA_PATH)
    print_key_value("Selected device", device)
    print_key_value("Checkpoint path", CHECKPOINT_PATH)

    X_ecg_val, X_cgm_val, y_val, data_source = load_validation_data()
    X_ecg_tensor, X_cgm_tensor, y_tensor = to_tensors(X_ecg_val, X_cgm_val, y_val)
    model, model_import_path = load_model(device)
    predictions = run_inference(model, X_ecg_tensor, X_cgm_tensor, device)
    y_true = y_tensor.detach().cpu().numpy()

    y_true_sys = y_true[:, 0]
    y_true_dia = y_true[:, 1]
    y_pred_sys = predictions[:, 0]
    y_pred_dia = predictions[:, 1]

    metrics = {
        "Systolic": calculate_metrics("Systolic", y_true_sys, y_pred_sys),
        "Diastolic": calculate_metrics("Diastolic", y_true_dia, y_pred_dia),
    }

    print_metrics_report(metrics, y_true.shape[0], device, model_import_path, data_source, y_true)
    save_metrics_outputs(metrics, y_true.shape[0], device, model_import_path, data_source)
    plot_actual_vs_predicted(y_true_sys, y_true_dia, y_pred_sys, y_pred_dia, metrics)
    plot_bland_altman(y_true_sys, y_true_dia, y_pred_sys, y_pred_dia, metrics)

    print_section("OUTPUT ARTEFACTS")
    print_key_value("Metrics CSV", METRICS_CSV_PATH)
    print_key_value("Metrics JSON", METRICS_JSON_PATH)
    print_key_value("Predicted vs Actual", ACTUAL_VS_PRED_PATH)
    print_key_value("Bland-Altman", BLAND_ALTMAN_PATH)
    print("\nEvaluation completed successfully.")


def main() -> int:
    try:
        run_evaluation()
        return 0
    except Exception as exc:
        print("\n" + "!" * 90)
        print("EVALUATION FAILED")
        print("!" * 90)
        print(f"{type(exc).__name__}: {exc}")
        print("\nTraceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
