# Multimodal Late-Fusion Deep Learning for Cuffless Blood Pressure Estimation

ECG and continuous glucose monitoring (CGM) as paired inputs to a late-fusion neural regressor for systolic and diastolic blood pressure, evaluated against ambulatory blood pressure monitor (ABPM) reference labels.

## Dataset

The recordings used in this project come from a publicly available Dryad dataset. The raw data is not redistributed in this repository.

- **DOI**: [10.5061/dryad.qjq2bvqpk](https://doi.org/10.5061/dryad.qjq2bvqpk)
- **Title**: 24-hour physiological monitoring; electrocardiogram, interstitial glucose, ambulatory blood pressure
- **Cohort**: 30 participants
- **Sampling**: Zephyr BioHarness ECG at 250 Hz; continuous interstitial glucose; sparse ambulatory blood pressure cuff readings

See `data/raw/README.md` for the list of files to download from Dryad if you intend to retrain the models.

## Repository layout

```
blood-pressure-estimation/
  src/
    models/
      ecg_branch.py        1D-ResNet feature extractor for ECG and ECG-only baseline
      cgm_branch.py        LSTM feature extractor for CGM history windows
      fusion.py            Multimodal regressor that combines both branches
    preprocessing/
      alignment.py         Time-window alignment of ECG, CGM, and ABPM streams
      data_loader.py       Per-subject file loaders
      filters.py           Bandpass filtering for raw ECG
    utils/
      metrics.py           Evaluation utilities
    train.py               Training loop for the fusion model
    evaluate.py            Validation pass for the fusion model
    evaluate_baseline.py   Validation pass for the ECG-only baseline
  notebooks/
    01_eda_ecg.ipynb       Exploratory data analysis of ECG signals for one subject
  models/
    checkpoints/           Pre-trained weights (best_fusion_model.pth, best_ecg_baseline.pth)
  results/
    cache/val_split.npz    Cached validation split (lets evaluate.py run without raw data)
    plots/                 Pre-rendered evaluation figures
    *.csv, *.json          Metrics for fusion and baseline models
  data/raw/README.md       Pointer to the Dryad dataset for the raw files
  requirements.txt         Python dependencies
  LICENSE                  MIT
  README.md                This file
```

## Quick start (no raw data required)

The repository ships the trained checkpoints, the cached validation split, and the pre-rendered plots, so the headline metrics can be reproduced without retraining and without downloading raw data:

```bash
git clone <this-repo-url>
cd blood-pressure-estimation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.evaluate
python -m src.evaluate_baseline
```

Both commands print a metrics table to standard output and overwrite the figures in `results/plots/`. They do not modify the cached split or the checkpoints.

## Full retraining

Retraining requires the full Dryad dataset placed under `data/raw/` according to the layout described in `data/raw/README.md`. With the data in place:

```bash
python -m src.train
```

This re-aligns every subject, then splits the aligned observations 80/20 between training and validation. It trains `MultiModalBPRegressor` for 30 epochs (default) on the available device (CUDA, then MPS, then CPU), saves the best validation-loss checkpoint to `models/checkpoints/best_fusion_model.pth`, and writes a training-curve plot to `results/plots/fusion_loss_curve.png`.

The baseline ECG-only checkpoint included in this repository was produced by an earlier training script that is not part of this codebase. The baseline can still be evaluated against the same cached validation split via `python -m src.evaluate_baseline`.

## Architecture

A late-fusion neural regressor that combines two parallel feature extractors:

- **ECG branch** (`src/models/ecg_branch.py`): a 1D residual convolutional network applied to a 10-second ECG window at 250 Hz, producing a 128-dimensional embedding. The same encoder is reused as a standalone baseline by attaching a linear regression head.
- **CGM branch** (`src/models/cgm_branch.py`): a 2-layer LSTM over a 30-minute glucose history window with paired derivative features, producing a 64-dimensional embedding.
- **Fusion head** (`src/models/fusion.py`): the two embeddings are concatenated and passed through a small MLP that outputs systolic and diastolic blood pressure jointly.

Training minimises L1 loss (mean absolute error), which is the metric specified by AAMI for blood pressure device evaluation. Optimisation uses AdamW with a `ReduceLROnPlateau` schedule on the validation MAE.

## Results

132-sample validation set, 80/20 random split of aligned observations:

| Channel   | Model             | MAE (mmHg) | RMSE (mmHg) | Pearson r |
| :-------- | :---------------- | ---------: | ----------: | --------: |
| Systolic  | Fusion            |      11.58 |       15.04 |      0.47 |
| Diastolic | Fusion            |       8.65 |       11.25 |      0.45 |
| Systolic  | ECG-only baseline |     113.20 |      114.49 |     -0.08 |
| Diastolic | ECG-only baseline |      65.75 |       66.95 |     -0.05 |

The ECG-only baseline did not converge to a useful regressor on this data partition; its predictions remain close to zero, which the metrics reflect as a near-constant negative bias. The collapsed baseline is reported here for transparency. The fusion model produces lower error than the baseline across both channels but does not meet established clinical accuracy standards for cuff-replacement use.

The directory `results/plots/` contains pre-rendered Bland-Altman agreement plots, predicted-versus-actual scatter plots, and the fusion training-loss curve.

## Limitations

- **Sample size**: 30 participants and 132 validation observations. The split is randomised at the observation level rather than at the subject level, so within-subject correlation may inflate the validation Pearson r.
- **CGM coverage**: not every participant in the Dryad cohort wore a CGM. Subjects without CGM are excluded from the multimodal training set, reducing the effective cohort.
- **Mode collapse on the baseline**: the ECG-only baseline weights produced near-constant predictions on the validation split. The fusion result therefore cannot be interpreted as evidence that CGM alone rescues a working ECG-only regressor.

## Citation

If you make use of this repository in academic work, please cite:

> Elaydi, M. (2026) *Multimodal Late-Fusion Deep Learning for Cuffless Blood Pressure Estimation: ECG and Continuous Glucose Monitoring against Ambulatory Reference Labels*. BEng Computer Systems Engineering dissertation, School of Electronic Engineering and Computer Science, Queen Mary University of London.

Please also cite the Dryad dataset as the data source.

## License

Released under the MIT License. See `LICENSE` for the full text.

## Acknowledgement

Final-year undergraduate project supervised at Queen Mary University of London, School of Electronic Engineering and Computer Science.
