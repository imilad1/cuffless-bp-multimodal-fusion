# Raw Data Directory

This directory is where the raw physiological recordings used to train and evaluate the models are expected to live. The recordings are not committed to this repository because of their size (about 41 GB) and because the dataset is already publicly archived elsewhere.

## Source

The dataset is publicly available from Dryad:

- **DOI**: [10.5061/dryad.qjq2bvqpk](https://doi.org/10.5061/dryad.qjq2bvqpk)
- **Title**: 24-hour physiological monitoring; electrocardiogram, interstitial glucose, ambulatory blood pressure
- **Cohort**: 30 participants
- **Modalities**:
  - ECG waveform at 250 Hz (Zephyr BioHarness)
  - Continuous interstitial glucose monitoring
  - Ambulatory blood pressure cuff readings (sparse)

Refer to the dataset documentation distributed with the Dryad archive for full schema and column-level descriptions.

## Files needed for retraining

After downloading the Dryad archive, place the following files into this directory (`data/raw/`) so that the preprocessing code can locate them:

- `Per_Participant_Sensor_Data.zip`, then extract its contents into `data/raw/extracted_sensor_data/`
- `Output_ECG_Segmentor_data.zip` (optional, extracted into `data/raw/extracted_segments/` if used)
- `Blood_Pressure_Sleep_Info.xlsx`
- `Data_Collection_Notes.csv`
- `Participant_Information.csv`

The default training script (`python -m src.train`) iterates over subjects `001` to `030` and skips any subject whose folder is missing or incomplete, so a partial download is tolerated, with the caveat that the final aligned dataset size shrinks accordingly.

## When raw data is not required

The repository ships:

- Trained model checkpoints in `models/checkpoints/`
- A pre-computed validation split in `results/cache/val_split.npz`
- Pre-rendered evaluation figures and metric tables in `results/`

These together let `python -m src.evaluate` and `python -m src.evaluate_baseline` reproduce the reported metrics without any raw data downloaded.
