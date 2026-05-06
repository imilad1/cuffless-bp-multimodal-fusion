"""
Data loader module for blood pressure estimation project.
Handles loading of ABPM, CGM, and ECG data from the raw dataset structure.
"""

from glob import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd


class DataLoader:
    """
    DataLoader for the blood pressure estimation dataset.
    
    Expected raw data structure:
    - ABPM: {ID}/{ID}_ABPM/{ID}_ABPM.csv
    - CGM: {ID}/{ID}_CGM/{ID}_glucose.csv
    - ECG: {ID}/{ID}_Zephyr/*_ECG.csv (multiple timestamped files)
    """
    
    def __init__(self, raw_data_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            raw_data_path: Path to the raw data directory containing
                           extracted_sensor_data/Per_Participant_Sensor_Data/
        """
        self.raw_data_path = Path(raw_data_path)
        self.sensor_data_path = self.raw_data_path / "extracted_sensor_data" / "Per_Participant_Sensor_Data"
        self.segments_path = self.raw_data_path / "extracted_segments"
        
        if not self.sensor_data_path.exists():
            raise FileNotFoundError(
                f"Sensor data path does not exist: {self.sensor_data_path}"
            )
    
    def get_abpm(self, subject_id: str) -> pd.DataFrame:
        """
        Load ABPM (Ambulatory Blood Pressure Monitoring) data for a subject.
        
        Supports both .csv and .xlsx/.xls file formats.
        
        Args:
            subject_id: Subject identifier (e.g., '001')
            
        Returns:
            DataFrame containing ABPM measurements
            
        Raises:
            FileNotFoundError: If the ABPM file does not exist in any format
        """
        abpm_dir = self.sensor_data_path / subject_id / f"{subject_id}_ABPM"
        csv_path = abpm_dir / f"{subject_id}_ABPM.csv"
        xlsx_path = abpm_dir / f"{subject_id}_ABPM.xlsx"
        xls_path = abpm_dir / f"{subject_id}_ABPM.xls"
        
        if csv_path.exists():
            return pd.read_csv(csv_path)
        elif xlsx_path.exists():
            return pd.read_excel(xlsx_path)
        elif xls_path.exists():
            return pd.read_excel(xls_path)
        else:
            raise FileNotFoundError(
                f"ABPM file not found for subject {subject_id} "
                f"(checked .csv, .xlsx, .xls in {abpm_dir})"
            )

    @staticmethod
    def _read_table(path: Path, header: Optional[int] = 0) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=header)
        return pd.read_excel(path, header=header)

    @staticmethod
    def _find_cgm_header_row(raw_df: pd.DataFrame) -> int:
        for idx, row in raw_df.iterrows():
            values = {
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value) and str(value).strip()
            }
            if "Device Timestamp" in values:
                return idx
        raise KeyError("Device Timestamp")

    @staticmethod
    def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> str:
        normalised = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in normalised:
                return normalised[key]
        raise KeyError(candidates[0])
    
    def get_cgm(self, subject_id: str) -> pd.DataFrame:
        """
        Load CGM (Continuous Glucose Monitoring) data for a subject.
        
        Supports both .csv and .xlsx/.xls file formats.
        
        Args:
            subject_id: Subject identifier (e.g., '001')
            
        Returns:
            DataFrame containing CGM measurements
            
        Raises:
            FileNotFoundError: If the CGM file does not exist in any format
        """
        cgm_dir = self.sensor_data_path / subject_id / f"{subject_id}_CGM"
        csv_path = cgm_dir / f"{subject_id}_glucose.csv"
        xlsx_path = cgm_dir / f"{subject_id}_glucose.xlsx"
        xls_path = cgm_dir / f"{subject_id}_glucose.xls"
        
        if csv_path.exists():
            cgm_path = csv_path
        elif xlsx_path.exists():
            cgm_path = xlsx_path
        elif xls_path.exists():
            cgm_path = xls_path
        else:
            raise FileNotFoundError(
                f"CGM file not found for subject {subject_id} "
                f"(checked .csv, .xlsx, .xls in {cgm_dir})"
            )

        raw_df = self._read_table(cgm_path, header=None)
        header_idx = self._find_cgm_header_row(raw_df)
        columns = raw_df.iloc[header_idx].fillna("").astype(str).str.strip()
        cgm_df = raw_df.iloc[header_idx + 1:].reset_index(drop=True).copy()
        cgm_df.columns = columns
        cgm_df = cgm_df.loc[:, [str(col).strip() != "" for col in cgm_df.columns]]
        cgm_df = cgm_df.dropna(how="all").reset_index(drop=True)
        cgm_df.columns = cgm_df.columns.astype(str).str.strip()

        timestamp_col = self._find_first_column(cgm_df, ["Device Timestamp"])
        glucose_col = self._find_first_column(
            cgm_df,
            [
                "Historic Glucose mmol/L",
                "Historic Glucose mg/dL",
                "Historic Glucose",
                "Glucose",
            ],
        )

        cgm_df["timestamp"] = pd.to_datetime(
            cgm_df[timestamp_col],
            dayfirst=True,
            errors="coerce",
        )
        cgm_df["glucose"] = pd.to_numeric(cgm_df[glucose_col], errors="coerce")

        valid_glucose = cgm_df.loc[
            cgm_df["glucose"].notna() & (cgm_df["glucose"] > 0),
            "glucose",
        ]
        median_glucose = float(valid_glucose.median()) if not valid_glucose.empty else float("nan")
        p90_glucose = float(valid_glucose.quantile(0.9)) if not valid_glucose.empty else float("nan")
        max_glucose = float(valid_glucose.max()) if not valid_glucose.empty else float("nan")
        converted_from_mgdl = bool(
            not valid_glucose.empty and (median_glucose > 30 or p90_glucose > 30)
        )

        if converted_from_mgdl:
            cgm_df["glucose"] = cgm_df["glucose"] / 18.0182

        cgm_df = cgm_df.dropna(subset=["timestamp", "glucose"])
        cgm_df = cgm_df.sort_values("timestamp").reset_index(drop=True)
        cgm_df.attrs["source_path"] = str(cgm_path)
        cgm_df.attrs["glucose_column"] = str(glucose_col)
        cgm_df.attrs["glucose_median_raw"] = median_glucose
        cgm_df.attrs["glucose_p90_raw"] = p90_glucose
        cgm_df.attrs["glucose_max_raw"] = max_glucose
        cgm_df.attrs["converted_from_mgdl"] = converted_from_mgdl
        cgm_df.attrs["glucose_unit"] = (
            "mg/dL converted to mmol/L" if converted_from_mgdl else "mmol/L"
        )
        return cgm_df
    
    def get_ecg_paths(self, subject_id: str) -> List[Path]:
        """
        Get all ECG file paths for a subject, sorted by filename.
        
        The ECG data is split across multiple timestamped files ending in _ECG.csv.
        Files are sorted by filename to maintain chronological order.
        
        Args:
            subject_id: Subject identifier (e.g., '001')
            
        Returns:
            List of Path objects pointing to ECG files, sorted by filename
            
        Raises:
            FileNotFoundError: If the Zephyr directory does not exist
        """
        zephyr_path = self.sensor_data_path / subject_id / f"{subject_id}_Zephyr"
        
        if not zephyr_path.exists():
            raise FileNotFoundError(f"Zephyr directory not found: {zephyr_path}")
        
        ecg_pattern = str(zephyr_path / "*_ECG.csv")
        ecg_files = glob(ecg_pattern)
        
        ecg_paths = sorted([Path(f) for f in ecg_files], key=lambda p: p.name)
        
        return ecg_paths
    
    def load_ecg(self, subject_id: str) -> List[pd.DataFrame]:
        """
        Load all ECG data files for a subject.
        
        Args:
            subject_id: Subject identifier (e.g., '001')
            
        Returns:
            List of DataFrames, one per ECG file, in chronological order
        """
        ecg_paths = self.get_ecg_paths(subject_id)
        return [pd.read_csv(path) for path in ecg_paths]
    
    def load_segments(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Load pre-segmented data for a subject from extracted_segments directory.
        
        Args:
            subject_id: Subject identifier (e.g., '001')
            
        Returns:
            DataFrame containing segmented data, or None if not found
        """
        if not self.segments_path.exists():
            raise FileNotFoundError(f"Segments path does not exist: {self.segments_path}")
        
        segment_pattern = str(self.segments_path / f"{subject_id}*")
        segment_files = glob(segment_pattern)
        
        if not segment_files:
            return None
        
        segment_path = Path(segment_files[0])
        
        if segment_path.suffix == '.csv':
            return pd.read_csv(segment_path)
        elif segment_path.suffix == '.parquet':
            return pd.read_parquet(segment_path)
        else:
            return pd.read_csv(segment_path)
    
    def get_available_subjects(self) -> List[str]:
        """
        Get list of all available subject IDs in the dataset.
        
        Returns:
            List of subject ID strings
        """
        subjects = []
        for item in self.sensor_data_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                subjects.append(item.name)
        return sorted(subjects)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_data_path = project_root / "data" / "raw"
    
    print(f"Initializing DataLoader with path: {raw_data_path}")
    loader = DataLoader(str(raw_data_path))
    
    subject_id = "001"
    print(f"\n{'='*60}")
    print(f"Testing data loading for Subject: {subject_id}")
    print(f"{'='*60}")
    
    try:
        ecg_paths = loader.get_ecg_paths(subject_id)
        print(f"\n[ECG] Number of ECG files found: {len(ecg_paths)}")
        if ecg_paths:
            print(f"[ECG] First file: {ecg_paths[0].name}")
            print(f"[ECG] Last file: {ecg_paths[-1].name}")
    except FileNotFoundError as e:
        print(f"[ECG] Error: {e}")
    
    try:
        abpm_df = loader.get_abpm(subject_id)
        print(f"\n[ABPM] Shape: {abpm_df.shape}")
        print(f"[ABPM] Columns: {list(abpm_df.columns)}")
        print(f"\n[ABPM] Head:")
        print(abpm_df.head())
    except FileNotFoundError as e:
        print(f"[ABPM] Error: {e}")
    
    try:
        cgm_df = loader.get_cgm(subject_id)
        print(f"\n[CGM] Shape: {cgm_df.shape}")
        print(f"[CGM] Columns: {list(cgm_df.columns)}")
        print(f"\n[CGM] Head:")
        print(cgm_df.head())
    except FileNotFoundError as e:
        print(f"[CGM] Error: {e}")
    
    print(f"\n{'='*60}")
    print("Data loading test complete.")
    print(f"{'='*60}")
