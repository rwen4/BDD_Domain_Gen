# Temporal Domain Validation and MIMIC-IV Temporal Dataset Pipeline

This project preprocesses MIMIC-IV hospital admissions into temporal domains and evaluates whether clinical distributions shift over time.

The pipeline:
1. Processes MIMIC-IV admissions into temporal datasets
2. Creates temporal train/test splits
3. Builds domain discriminator models
4. Measures how distinguishable time periods are from patient features

The goal is to validate whether temporal drift exists strongly enough to justify predictive domain adaptation methods such as Kalman filtering.

---

## Files

| File | Purpose |
|---|---|
| `mimic_iv_Version2.py` | MIMIC-IV preprocessing and temporal dataset generation |
| `domain_validation.py` | Domain discriminator for temporal shift validation |

---

# 1. MIMIC-IV Temporal Dataset Generation

## Overview

`mimic_iv_Version2.py` processes:
- admissions
- patients
- diagnoses_icd

from MIMIC-IV into temporal prediction datasets.

The pipeline creates:
- 3-year temporal windows
- mortality labels
- diagnosis representations
- train/test splits

---

## Temporal Domains

The dataset is split into:

| Domain | Years |
|---|---|
| 2008 | 2008–2010 |
| 2011 | 2011–2013 |
| 2014 | 2014–2016 |
| 2017 | 2017–2019 |
| 2020 | 2020–2022 |

These are stored in:

```python
TIME_PERIODS = [2008, 2011, 2014, 2017, 2020]
```

---

## Features

Each admission contains:
- Age
- Gender
- Ethnicity/race
- ICD diagnosis sequences

Diagnoses are converted into:
- concatenated ICD-code strings
- temporal feature representations

---

## Outcome

The default task is:
- in-hospital mortality prediction

using:
- `hospital_expire_flag`

---

## Processing Output

The preprocessing pipeline generates:
- `mimic_iv_processed.csv`
- `mimic_iv_stay_dict.pkl`
- `mimic_iv_wildtime.pkl`

---

## Running Preprocessing

```bash
python mimic_iv_Version2.py /path/to/mimiciv/
```

Force rebuilding:

```bash
python mimic_iv_Version2.py /path/to/mimiciv/ --force
```

---

# 2. Domain Validation

## Goal

`domain_validation.py` tests whether patient features contain enough temporal information to identify the admission period.

This is done using a domain discriminator:

\[
D(X) \rightarrow T
\]

where:
- \(X\) = patient features
- \(T\) = temporal domain

If the discriminator predicts domains substantially above random chance, temporal drift is confirmed.

---

## Domain Discriminator Architecture

The discriminator:
- encodes ICD codes into multi-hot vectors
- combines demographic features
- trains a neural network classifier to predict time period

Architecture:
- fully connected MLP
- batch normalization
- dropout regularization

---

## Why This Matters

If time periods are distinguishable:
- the dataset exhibits temporal distribution shift
- temporal signatures exist in the feature space
- predictive adaptation methods become justified

If performance is near random:
- domains may be approximately stationary

---

## Running Domain Validation

```bash
python domain_validation.py /path/to/mimiciv/
```

Optional:

```bash
python domain_validation.py /path/to/mimiciv/ --max-samples 50000
```

---

## Outputs

The script reports:
- domain classification accuracy
- random baseline accuracy
- temporal distinguishability
- average domain probability distributions

---

## Dependencies

Install:

```bash
pip install numpy pandas torch scikit-learn
```

---
