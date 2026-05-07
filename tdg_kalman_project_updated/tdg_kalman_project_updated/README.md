# Temporal Domain Generalization + Kalman Adaptation Project

This project contains code for constructing MIMIC-IV temporal domain generalization datasets, analyzing temporal drift, and testing Kalman-guided predictive adaptation.

## Recommended workflow

### 1. Create datasets

Run `dataset_creation.py` first. It expects the following MIMIC-IV files under:

```text
~/scratch/mimiciv_wildtime_data/
```

Required files:

```text
patients.csv.gz
admissions.csv.gz
diagnoses_icd.csv.gz
procedures_icd.csv.gz
```

Command:

```bash
python scripts/dataset_creation.py
```

By default, it saves outputs to:

```text
~/scratch/temporal_dg_outputs_noleak/
```

It generates three datasets:

```text
dataset_a_demo_dx_nocad.csv          # demographics + non-CAD diagnosis features
dataset_b_demo_dx_proc_nocad.csv     # demographics + non-CAD diagnosis features + procedures
dataset_c_demo_proc.csv              # demographics + procedures only
```

The CAD label is created from ICD-10 I20-I25 and ICD-9 410-414 codes. CAD-defining diagnosis codes are removed from diagnosis features to reduce direct label leakage.

### 2. Analyze temporal drift/generalization

For one dataset, especially Dataset B:

```bash
python scripts/datasetb_analysis.py \
  --dataset_path ~/scratch/temporal_dg_outputs_noleak/dataset_b_demo_dx_proc_nocad.csv \
  --out_dir outputs/dataset_b_analysis
```

For all `dataset_*.csv` files in a directory:

```bash
python scripts/temporal_dg_analysis.py \
  --input_dir ~/scratch/temporal_dg_outputs_noleak \
  --out_dir outputs/temporal_analysis
```

Note: `temporal_dg_analysis.py` currently searches for files matching `dataset_*.csv`. If your generated files are named `dataset_a_demo_dx_nocad.csv`, etc., this works.

### 3. Run Kalman-guided adaptation

```bash
python scripts/kalman_initial.py \
  --dataset_path ~/scratch/temporal_dg_outputs_noleak/dataset_b_demo_dx_proc_nocad.csv \
  --out_dir outputs/kalman_dataset_b \
  --device cpu
```

If using a GPU on the cluster:

```bash
python scripts/kalman_initial.py \
  --dataset_path ~/scratch/temporal_dg_outputs_noleak/dataset_b_demo_dx_proc_nocad.csv \
  --out_dir outputs/kalman_dataset_b \
  --device cuda
```

## Main scripts

```text
scripts/dataset_creation.py       Build datasets A/B/C from raw MIMIC-IV tables
scripts/datasetb_analysis.py      Single-dataset temporal generalization and drift analysis
scripts/temporal_dg_analysis.py   Multi-dataset temporal generalization and drift analysis
scripts/kalman_initial.py         Kalman-guided predictive adaptation experiment
```

## Removed/archived

`pcavisualization.py` was intentionally removed because it was a one-off PCA visualization script with hard-coded paths and an older label convention.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Key dependencies:

```text
numpy
pandas
matplotlib
scikit-learn
scipy
pykalman
torch
```
