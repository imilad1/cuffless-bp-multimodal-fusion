"""
Logic to synchronize high-freq ECG (250Hz) with low-freq CGM (5min) and sparse ABPM.

Implements strict time-window extraction so that every aligned sample is
backed by real sensor data, no zero-padding, no interpolation gaps.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeAligner:
    """Fuse 250 Hz ECG and 5-min CGM windows against sparse ABPM timestamps.

    For each valid ABPM reading the aligner extracts:
      - A fixed-length ECG waveform window ending at the BP measurement time.
      - A CGM history window (glucose values + rate-of-change) preceding the
        BP measurement time.

    Readings are **silently dropped** when either sensor window cannot be
    fully satisfied (sensor gap, file boundary, etc.).

    Args:
        ecg_window_sec: Seconds of ECG to extract per BP reading.
        cgm_window_min: Minutes of CGM history to extract per BP reading.
        ecg_hz: ECG sampling rate in Hz.
    """

    # Minimum fraction of expected CGM readings required to accept a window.
    _CGM_MIN_READINGS: int = 2

    def __init__(
        self,
        ecg_window_sec: int = 30,
        cgm_window_min: int = 30,
        ecg_hz: int = 250,
    ) -> None:
        self.ecg_window_sec: int = ecg_window_sec
        self.cgm_window_min: int = cgm_window_min
        self.ecg_hz: int = ecg_hz
        self.required_ecg_samples: int = self.ecg_window_sec * self.ecg_hz
        self.max_cgm_readings: int = (self.cgm_window_min // 5) + 1
        self._ecg_cache: Dict[Path, pd.DataFrame] = {}

        logger.info(
            "TimeAligner initialised: ecg_window=%ds (%d samples @ %dHz), "
            "cgm_window=%dmin",
            self.ecg_window_sec,
            self.required_ecg_samples,
            self.ecg_hz,
            self.cgm_window_min,
        )

    # CGM extraction
    def _extract_cgm_window(
        self,
        cgm_df: pd.DataFrame,
        target_time: datetime,
    ) -> Optional[np.ndarray]:
        """Extract a CGM window of glucose + derivative preceding *target_time*.

        Args:
            cgm_df: DataFrame with a ``timestamp`` column (datetime) and a
                ``glucose`` column (float).  Must be sorted by timestamp.
            target_time: The ABPM measurement time that anchors the window.

        Returns:
            A 2-D numpy array of shape ``(n_readings, 2)`` where column 0 is
            glucose and column 1 is the glucose rate-of-change, or ``None``
            if the window has fewer than ``_CGM_MIN_READINGS`` readings.
        """
        window_start: datetime = target_time - timedelta(minutes=self.cgm_window_min)

        mask = (cgm_df["timestamp"] >= window_start) & (
            cgm_df["timestamp"] <= target_time
        )
        window: pd.DataFrame = cgm_df.loc[mask].copy()

        if len(window) < self._CGM_MIN_READINGS:
            logger.debug(
                "CGM window rejected: %d readings (need >= %d) at %s",
                len(window),
                self._CGM_MIN_READINGS,
                target_time,
            )
            return None

        glucose: pd.Series = window["glucose"].astype(float)
        glucose = glucose.ffill()  # forward-fill internal NaNs

        derivative: pd.Series = glucose.diff()
        derivative.iloc[0] = 0.0  # first derivative is undefined; set to 0

        result: np.ndarray = np.column_stack(
            [glucose.values, derivative.values]
        )

        # Pad or truncate to fixed length for uniform tensor shape
        padded: np.ndarray = np.zeros((self.max_cgm_readings, 2), dtype=np.float64)
        n: int = min(result.shape[0], self.max_cgm_readings)
        padded[:n, :] = result[:n, :]
        return padded

    # ECG extraction
    @staticmethod
    def _parse_ecg_start_time(ecg_path: Path) -> datetime:
        """Derive the recording start time from the ECG filename.

        Expected pattern: ``YYYY_MM_DD__HH_MM_SS_ECG.csv``

        Args:
            ecg_path: Path to the ECG CSV file.

        Returns:
            A timezone-naive datetime.
        """
        stem: str = ecg_path.stem.replace("_ECG", "")
        date_part, time_part = stem.split("__")
        return datetime.strptime(f"{date_part}_{time_part}", "%Y_%m_%d_%H_%M_%S")

    def _build_ecg_index(
        self,
        ecg_paths: List[Path],
    ) -> List[Dict[str, object]]:
        """Build a lightweight time index for every ECG file.

        Each entry stores the file path, start time, and end time so that
        ``_extract_ecg_window`` can locate the correct file without loading
        every CSV upfront.

        Args:
            ecg_paths: Sorted list of ECG CSV paths.

        Returns:
            A list of dicts with keys ``path``, ``start``, ``end``.
        """
        index: List[Dict[str, object]] = []
        for path in ecg_paths:
            start_time: datetime = self._parse_ecg_start_time(path)
            # Read only the Time column to get end time efficiently
            time_col: pd.Series = pd.read_csv(path, usecols=["Time"])["Time"]
            end_time: datetime = pd.to_datetime(
                time_col.iloc[-1],
                dayfirst=True,
                errors="coerce",
            )
            n_samples: int = len(time_col)
            index.append(
                {
                    "path": path,
                    "start": start_time,
                    "end": end_time,
                    "n_samples": n_samples,
                }
            )
            logger.debug(
                "ECG index: %s  %s to %s  (%d samples)",
                path.name,
                start_time,
                end_time,
                n_samples,
            )
        return index

    def _extract_ecg_window(
        self,
        ecg_index: List[Dict[str, object]],
        target_time: datetime,
    ) -> Optional[np.ndarray]:
        """Extract a fixed-length ECG waveform ending at *target_time*.

        Args:
            ecg_index: Pre-built index from ``_build_ecg_index``.
            target_time: The ABPM measurement time that anchors the window.

        Returns:
            A 1-D numpy array of shape ``(required_ecg_samples,)`` containing
            the raw waveform, or ``None`` if no file covers the window or the
            window is shorter than required.
        """
        window_start: datetime = target_time - timedelta(seconds=self.ecg_window_sec)

        # Find the file whose time range contains the window
        target_entry: Optional[Dict[str, object]] = None
        for entry in ecg_index:
            if entry["start"] <= window_start and entry["end"] >= target_time:
                target_entry = entry
                break

        if target_entry is None:
            logger.debug(
                "ECG window rejected: no file covers %s to %s",
                window_start,
                target_time,
            )
            return None

        # Load the full file once and cache per path to avoid re-reading
        # the same multi-million-row CSV for every ABPM target.
        ecg_path: Path = target_entry["path"]
        ecg_df: Optional[pd.DataFrame] = self._ecg_cache.get(ecg_path)
        if ecg_df is None:
            ecg_df = pd.read_csv(ecg_path)
            ecg_df["Time"] = pd.to_datetime(
                ecg_df["Time"],
                dayfirst=True,
                errors="coerce",
            )
            self._ecg_cache[ecg_path] = ecg_df

        # Find the closest index to target_time and slice backward
        end_idx: int = (ecg_df["Time"] - target_time).abs().idxmin()
        start_idx: int = end_idx - self.required_ecg_samples + 1

        if start_idx < 0:
            logger.debug(
                "ECG window rejected: slice starts at %d (need %d samples) in %s",
                start_idx,
                self.required_ecg_samples,
                target_entry["path"].name,
            )
            return None

        waveform: np.ndarray = (
            ecg_df.loc[start_idx:end_idx, "EcgWaveform"]
            .to_numpy(dtype=np.float64)
        )

        if waveform.shape[0] != self.required_ecg_samples:
            logger.debug(
                "ECG window rejected: got %d samples, need %d",
                waveform.shape[0],
                self.required_ecg_samples,
            )
            return None

        return waveform

    # Main synchronisation loop
    def align_subject(
        self,
        abpm_df: pd.DataFrame,
        cgm_df: pd.DataFrame,
        ecg_paths: List[Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align all three sensor streams for one subject.

        Args:
            abpm_df: ABPM DataFrame with columns ``timestamp`` (datetime),
                ``Systolic`` (numeric), and ``Diastolic`` (numeric).
                Only rows where both BP values are present and non-zero
                should be passed in.
            cgm_df: CGM DataFrame with columns ``timestamp`` (datetime) and
                ``glucose`` (float).
            ecg_paths: Sorted list of ECG CSV file paths.

        Returns:
            A tuple ``(X_ecg, X_cgm, y_bp)`` where:
              - **X_ecg**: ``np.ndarray`` of shape ``(N, required_ecg_samples)``
              - **X_cgm**: ``np.ndarray`` of shape ``(N, n_cgm_readings, 2)``
              - **y_bp**:  ``np.ndarray`` of shape ``(N, 2)``: [systolic, diastolic]

            ``N`` is the number of ABPM readings for which **both** sensor
            windows were fully satisfied.

        Raises:
            ValueError: If the input DataFrames lack required columns.
        """
        # ---- Input validation ----
        for col in ("timestamp", "Systolic", "Diastolic"):
            if col not in abpm_df.columns:
                raise ValueError(f"abpm_df missing required column: '{col}'")
        for col in ("timestamp", "glucose"):
            if col not in cgm_df.columns:
                raise ValueError(f"cgm_df missing required column: '{col}'")

        logger.info(
            "Starting alignment: %d ABPM readings, %d CGM readings, %d ECG files",
            len(abpm_df),
            len(cgm_df),
            len(ecg_paths),
        )

        # Pre-build ECG file index (reads only Time column per file)
        ecg_index: List[Dict[str, object]] = self._build_ecg_index(ecg_paths)
        self._ecg_cache.clear()

        X_ecg: List[np.ndarray] = []
        X_cgm: List[np.ndarray] = []
        y_bp: List[List[float]] = []
        skipped: int = 0

        for _, row in abpm_df.iterrows():
            target_time: datetime = row["timestamp"]
            systolic: float = float(row["Systolic"])
            diastolic: float = float(row["Diastolic"])

            cgm_window: Optional[np.ndarray] = self._extract_cgm_window(
                cgm_df, target_time
            )
            if cgm_window is None:
                skipped += 1
                continue

            ecg_window: Optional[np.ndarray] = self._extract_ecg_window(
                ecg_index, target_time
            )
            if ecg_window is None:
                skipped += 1
                continue

            X_ecg.append(ecg_window)
            X_cgm.append(cgm_window)
            y_bp.append([systolic, diastolic])

        logger.info(
            "Alignment complete: %d valid samples, %d skipped out of %d total",
            len(y_bp),
            skipped,
            len(abpm_df),
        )

        if len(y_bp) == 0:
            return (
                np.empty((0, self.required_ecg_samples)),
                np.empty((0, 0, 2)),
                np.empty((0, 2)),
            )

        return (
            np.array(X_ecg, dtype=np.float64),
            np.array(X_cgm, dtype=np.float64),
            np.array(y_bp, dtype=np.float64),
        )


# Standalone validation
if __name__ == "__main__":
    from datetime import datetime as dt

    from data_loader import DataLoader

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = project_root / "data" / "raw"

    loader = DataLoader(str(raw_data_path))
    subject_id: str = "001"

    print(f"\n{'=' * 60}")
    print(f"  TimeAligner validation: Subject {subject_id}")
    print(f"{'=' * 60}\n")

    # ---- Load ABPM ----
    abpm_df: pd.DataFrame = loader.get_abpm(subject_id)

    # Parse ABPM timestamps (same logic as notebook cell 3)
    ecg_paths: List[Path] = loader.get_ecg_paths(subject_id)
    first_ecg_start: datetime = TimeAligner._parse_ecg_start_time(ecg_paths[0])
    base_date = first_ecg_start.date()

    def _parse_abpm_time(time_str: str, base: object) -> Optional[datetime]:
        try:
            time_str = str(time_str).strip()
            time_part = time_str.split()[0]
            hour, minute = map(int, time_part.split(":"))
            return dt.combine(base, dt.min.time().replace(hour=hour, minute=minute))
        except Exception:
            return None

    abpm_df["timestamp"] = abpm_df["Time"].apply(
        lambda x: _parse_abpm_time(x, base_date)
    )

    # Adjust dates for midnight wrap-around
    prev_time = None
    date_offset = timedelta(days=0)
    timestamps: List[Optional[datetime]] = []
    for _, row in abpm_df.iterrows():
        ts = row["timestamp"]
        if ts is not None:
            if prev_time is not None and ts.hour < prev_time.hour and prev_time.hour > 20:
                date_offset += timedelta(days=1)
            ts = ts + date_offset
            prev_time = row["timestamp"]
        timestamps.append(ts)
    abpm_df["timestamp"] = timestamps

    # Filter valid BP readings
    abpm_valid: pd.DataFrame = abpm_df[
        (abpm_df["Systolic"].str.strip() != "0") & (abpm_df["timestamp"].notna())
    ].copy()
    abpm_valid["Systolic"] = pd.to_numeric(
        abpm_valid["Systolic"].str.strip(), errors="coerce"
    )
    abpm_valid["Diastolic"] = pd.to_numeric(
        abpm_valid["Diastolic"].str.strip(), errors="coerce"
    )
    abpm_valid = abpm_valid.dropna(subset=["Systolic", "Diastolic"])
    print(f"[ABPM]  Valid readings: {len(abpm_valid)}")

    # ---- Load CGM ----
    cgm_df: pd.DataFrame = loader.get_cgm(subject_id)
    print(f"[CGM]   Readings: {len(cgm_df)}")

    # ---- Align ----
    # Use 10-sec ECG window for quick testing (2500 samples instead of 7500)
    aligner = TimeAligner(ecg_window_sec=10, cgm_window_min=30, ecg_hz=250)
    X_ecg, X_cgm, y_bp = aligner.align_subject(abpm_valid, cgm_df, ecg_paths)

    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"{'=' * 60}")
    print(f"  X_ecg  shape : {X_ecg.shape}")
    print(f"  X_cgm  shape : {X_cgm.shape}")
    print(f"  y_bp   shape : {y_bp.shape}")
    if y_bp.shape[0] > 0:
        print(f"\n  First BP target  : {y_bp[0]} mmHg")
        print(f"  Last  BP target  : {y_bp[-1]} mmHg")
    print(f"{'=' * 60}\n")