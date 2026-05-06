"""
Main training loop. Orchestrates data loading, model forward pass, and backpropagation.

Phase IV - Multi-Modal Fusion:
  1. Aggregates aligned ECG + CGM + BP triples across all available subjects.
  2. Trains MultiModalBPRegressor (1D-ResNet + LSTM -> late fusion MLP)
     with MAE loss (AAMI clinical standard).
  3. Saves best checkpoint and loss curves.
"""

import sys
import re
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

# Ensure src/ is on the path so preprocessing / models are importable
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.fusion import MultiModalBPRegressor
from preprocessing.alignment import TimeAligner
from preprocessing.data_loader import DataLoader as RawDataLoader

# Hardware device
if torch.cuda.is_available():
    DEVICE: torch.device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[DEVICE] Using: {DEVICE}")


# PyTorch Dataset

class MultiModalDataset(Dataset):
    """Wraps aligned ECG + CGM windows and BP targets for PyTorch training.

    Applies per-sample Z-score normalisation to each ECG window so that
    the ResNet receives zero-mean, unit-variance input regardless of the
    raw amplitude scale across subjects/files.  CGM features (glucose +
    derivative) are already engineered by the TimeAligner and stored as-is.

    Args:
        X_ecg: Array of shape ``(N, seq_len)``.
        X_cgm: Array of shape ``(N, 7, 2)``.
        y_bp:  Array of shape ``(N, 2)``: [Systolic, Diastolic].
    """

    def __init__(self, X_ecg: np.ndarray, X_cgm: np.ndarray, y_bp: np.ndarray) -> None:
        # Per-sample Z-score normalisation (ECG only)
        means: np.ndarray = X_ecg.mean(axis=1, keepdims=True)
        stds: np.ndarray = X_ecg.std(axis=1, keepdims=True)
        stds[stds == 0] = 1.0  # guard against constant signals
        self.X_ecg: np.ndarray = ((X_ecg - means) / stds).astype(np.float32)
        self.X_cgm: np.ndarray = X_cgm.astype(np.float32)
        self.y_bp: np.ndarray = y_bp.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y_bp)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ecg: torch.Tensor = torch.from_numpy(self.X_ecg[idx]).unsqueeze(0)  # (1, seq_len)
        cgm: torch.Tensor = torch.from_numpy(self.X_cgm[idx])              # (7, 2)
        bp: torch.Tensor = torch.from_numpy(self.y_bp[idx])                # (2,)
        return ecg, cgm, bp


# Per-subject preprocessing helpers (mirrors alignment.py __main__)

def _parse_abpm_clock(value: object) -> Optional[time]:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.time().replace(microsecond=0)
    if hasattr(value, "hour") and hasattr(value, "minute"):
        try:
            return datetime.min.time().replace(
                hour=int(value.hour),
                minute=int(value.minute),
                second=int(getattr(value, "second", 0)),
            )
        except Exception:
            return None

    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(value).strip())
    if match is None:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    try:
        return datetime.min.time().replace(hour=hour, minute=minute, second=second)
    except ValueError:
        return None


def _build_abpm_timestamps(
    clocks: List[Optional[time]],
    base_date: object,
) -> List[Optional[datetime]]:
    timestamps: List[Optional[datetime]] = []
    day_offset = timedelta(days=0)
    prev_ts: Optional[datetime] = None

    for clock in clocks:
        if clock is None:
            timestamps.append(None)
            continue

        ts = datetime.combine(base_date, clock) + day_offset
        while prev_ts is not None and ts < prev_ts:
            day_offset += timedelta(days=1)
            ts = datetime.combine(base_date, clock) + day_offset

        timestamps.append(ts)
        prev_ts = ts

    return timestamps


def _infer_abpm_timestamps(
    clocks: List[Optional[time]],
    reference_ranges: List[Tuple[Optional[datetime], Optional[datetime]]],
) -> List[Optional[datetime]]:
    candidate_dates = set()
    for start, end in reference_ranges:
        for ref in (start, end):
            if ref is not None and not pd.isna(ref):
                for offset in range(-2, 3):
                    candidate_dates.add((ref + timedelta(days=offset)).date())

    if not candidate_dates:
        return [None for _ in clocks]

    valid_ranges = [
        (start, end)
        for start, end in reference_ranges
        if start is not None and end is not None and not pd.isna(start) and not pd.isna(end)
    ]

    def _score(base_date: object) -> Tuple[int, int, float]:
        timestamps = [ts for ts in _build_abpm_timestamps(clocks, base_date) if ts is not None]
        if not timestamps:
            return (-1, -1, float("-inf"))

        latest_start = max((start for start, _ in valid_ranges), default=None)
        earliest_end = min((end for _, end in valid_ranges), default=None)
        if latest_start is not None and earliest_end is not None and latest_start <= earliest_end:
            overlap_hits = sum(latest_start <= ts <= earliest_end for ts in timestamps)
        else:
            overlap_hits = 0

        range_hits = sum(
            start <= ts <= end
            for ts in timestamps
            for start, end in valid_ranges
        )
        if valid_ranges:
            coverage_start = min(start for start, _ in valid_ranges)
            coverage_end = max(end for _, end in valid_ranges)
            coverage_mid = coverage_start + (coverage_end - coverage_start) / 2
            median_ts = timestamps[len(timestamps) // 2]
            distance = abs((median_ts - coverage_mid).total_seconds())
        else:
            distance = 0.0

        return (overlap_hits, range_hits, -distance)

    best_date = max(candidate_dates, key=_score)
    return _build_abpm_timestamps(clocks, best_date)


def _get_ecg_range(ecg_paths: List[Path]) -> Tuple[Optional[datetime], Optional[datetime]]:
    if not ecg_paths:
        return None, None

    start_time = TimeAligner._parse_ecg_start_time(ecg_paths[0])
    end_time: Optional[datetime] = TimeAligner._parse_ecg_start_time(ecg_paths[-1])

    try:
        time_col = pd.read_csv(ecg_paths[-1], usecols=["Time"])["Time"]
        parsed_end = pd.to_datetime(time_col.iloc[-1], dayfirst=True, errors="coerce")
        if pd.notna(parsed_end):
            end_time = parsed_end.to_pydatetime()
    except Exception:
        pass

    return start_time, end_time


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce",
    )


def _prepare_subject(
    loader: RawDataLoader,
    aligner: TimeAligner,
    subject_id: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load, parse, align one subject's data. Returns (X_ecg, X_cgm, y_bp) or None."""

    # ---- ECG paths & base date ----
    ecg_paths: List[Path] = loader.get_ecg_paths(subject_id)
    if not ecg_paths:
        return None
    ecg_range = _get_ecg_range(ecg_paths)

    cgm_df: pd.DataFrame = loader.get_cgm(subject_id)
    cgm_range = (
        cgm_df["timestamp"].min().to_pydatetime() if not cgm_df.empty else None,
        cgm_df["timestamp"].max().to_pydatetime() if not cgm_df.empty else None,
    )

    # ---- ABPM ----
    abpm_df: pd.DataFrame = loader.get_abpm(subject_id)
    clocks = [_parse_abpm_clock(value) for value in abpm_df["Time"].tolist()]
    abpm_df["timestamp"] = _infer_abpm_timestamps(clocks, [ecg_range, cgm_range])

    # Filter valid BP readings
    abpm_df["Systolic"] = _clean_numeric_series(abpm_df["Systolic"])
    abpm_df["Diastolic"] = _clean_numeric_series(abpm_df["Diastolic"])
    abpm_valid: pd.DataFrame = abpm_df[
        (abpm_df["Systolic"] != 0) & (abpm_df["timestamp"].notna())
    ].copy()
    abpm_valid = abpm_valid.dropna(subset=["Systolic", "Diastolic"])

    if abpm_valid.empty:
        return None

    # ---- Align ----
    X_ecg, X_cgm, y_bp = aligner.align_subject(abpm_valid, cgm_df, ecg_paths)

    if y_bp.shape[0] == 0:
        return None

    return X_ecg, X_cgm, y_bp


# Global data aggregation

def prepare_global_dataset(
    raw_data_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterate over subjects 001 to 030, align data, and split into train/val.

    Args:
        raw_data_path: Path to ``data/raw/``.

    Returns:
        ``(X_ecg_train, X_ecg_val, X_cgm_train, X_cgm_val, y_train, y_val)``.
    """
    loader = RawDataLoader(str(raw_data_path))
    aligner = TimeAligner(ecg_window_sec=10, cgm_window_min=30, ecg_hz=250)

    all_X_ecg: List[np.ndarray] = []
    all_X_cgm: List[np.ndarray] = []
    all_y_bp: List[np.ndarray] = []

    for i in range(1, 31):
        subject_id: str = f"{i:03d}"
        try:
            result = _prepare_subject(loader, aligner, subject_id)
            if result is not None:
                X_ecg, X_cgm, y_bp = result
                all_X_ecg.append(X_ecg)
                all_X_cgm.append(X_cgm)
                all_y_bp.append(y_bp)
                print(f"  [Subject {subject_id}]  {y_bp.shape[0]} aligned samples")
            else:
                print(f"  [Subject {subject_id}]  skipped (no valid aligned data)")
        except FileNotFoundError as exc:
            print(f"  [Subject {subject_id}]  skipped ({exc})")
        except Exception as exc:
            print(f"  [Subject {subject_id}]  ERROR - {type(exc).__name__}: {exc}")

    if not all_X_ecg:
        raise RuntimeError("No aligned data found across any subject.")

    X_ecg_all: np.ndarray = np.vstack(all_X_ecg)
    X_cgm_all: np.ndarray = np.vstack(all_X_cgm)
    y_bp_all: np.ndarray = np.vstack(all_y_bp)
    print(f"\n[GLOBAL] Total aligned samples: {X_ecg_all.shape[0]}")
    print(f"[GLOBAL] X_ecg shape: {X_ecg_all.shape}  |  X_cgm shape: {X_cgm_all.shape}  |  y_bp shape: {y_bp_all.shape}")

    X_ecg_train, X_ecg_val, X_cgm_train, X_cgm_val, y_train, y_val = train_test_split(
        X_ecg_all, X_cgm_all, y_bp_all, test_size=0.2, random_state=42,
    )
    print(f"[SPLIT]  Train: {X_ecg_train.shape[0]}  |  Val: {X_ecg_val.shape[0]}\n")
    return X_ecg_train, X_ecg_val, X_cgm_train, X_cgm_val, y_train, y_val


# Training engine

def train_fusion(
    num_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> None:
    """Train the multi-modal fusion model end-to-end.

    Args:
        num_epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for AdamW.
        weight_decay: L2 regularisation strength.
    """
    raw_data_path: Path = _PROJECT_ROOT / "data" / "raw"

    # ---- Data ----
    print("=" * 60)
    print("  Phase 1: Data Aggregation & Alignment")
    print("=" * 60)
    X_ecg_train, X_ecg_val, X_cgm_train, X_cgm_val, y_train, y_val = (
        prepare_global_dataset(raw_data_path)
    )

    train_ds = MultiModalDataset(X_ecg_train, X_cgm_train, y_train)
    val_ds = MultiModalDataset(X_ecg_val, X_cgm_val, y_val)

    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- Model / Loss / Optimiser ----
    model: nn.Module = MultiModalBPRegressor().to(DEVICE)
    criterion: nn.Module = nn.L1Loss()  # MAE (AAMI clinical standard)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=3, factor=0.5)

    # ---- Checkpoint directory ----
    ckpt_dir: Path = _PROJECT_ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path: Path = ckpt_dir / "best_fusion_model.pth"

    # ---- Tracking ----
    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss: float = float("inf")

    print("=" * 60)
    print("  Phase 2: Training")
    print("=" * 60)
    print(f"  Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs       : {num_epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  LR           : {lr}")
    print(f"  Device       : {DEVICE}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        # ---- Train phase ----
        model.train()
        running_loss: float = 0.0
        n_train: int = 0

        for ecg_batch, cgm_batch, y_batch in train_loader:
            ecg_batch = ecg_batch.to(DEVICE)
            cgm_batch = cgm_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimiser.zero_grad()
            preds: torch.Tensor = model(ecg_batch, cgm_batch)
            loss: torch.Tensor = criterion(preds, y_batch)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * ecg_batch.size(0)
            n_train += ecg_batch.size(0)

        train_mae: float = running_loss / max(n_train, 1)
        train_losses.append(train_mae)

        # ---- Validation phase ----
        model.eval()
        running_val: float = 0.0
        n_val: int = 0

        with torch.no_grad():
            for ecg_batch, cgm_batch, y_batch in val_loader:
                ecg_batch = ecg_batch.to(DEVICE)
                cgm_batch = cgm_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(ecg_batch, cgm_batch)
                loss = criterion(preds, y_batch)

                running_val += loss.item() * ecg_batch.size(0)
                n_val += ecg_batch.size(0)

        val_mae: float = running_val / max(n_val, 1)
        val_losses.append(val_mae)

        scheduler.step(val_mae)

        # ---- Checkpointing ----
        ckpt_marker: str = ""
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            torch.save(model.state_dict(), best_ckpt_path)
            ckpt_marker = " * saved"

        current_lr: float = optimiser.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:3d}/{num_epochs}  |  "
            f"Train MAE: {train_mae:.4f}  |  "
            f"Val MAE: {val_mae:.4f}  |  "
            f"LR: {current_lr:.2e}{ckpt_marker}"
        )

    print("=" * 60)
    print(f"  Training complete. Best Val MAE: {best_val_loss:.4f} mmHg")
    print(f"  Checkpoint: {best_ckpt_path}")
    print("=" * 60)

    # ---- Loss curve ----
    plot_dir: Path = _PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path: Path = plot_dir / "fusion_loss_curve.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, num_epochs + 1)
    ax.plot(epochs_range, train_losses, label="Train MAE", linewidth=2)
    ax.plot(epochs_range, val_losses, label="Val MAE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("Multi-Modal Fusion: Train vs Validation Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved to: {plot_path}\n")


# Entry point
if __name__ == "__main__":
    train_fusion()