import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path.home() / "scratch" / "mimiciv_wildtime_data"
OUT_DIR = Path.home() / "scratch" / "temporal_dg_outputs_noleak"

PATIENTS_PATH = BASE_DIR / "patients.csv.gz"
ADMISSIONS_PATH = BASE_DIR / "admissions.csv.gz"
DIAGNOSES_PATH = BASE_DIR / "diagnoses_icd.csv.gz"
PROCEDURES_PATH = BASE_DIR / "procedures_icd.csv.gz"

LABEL_NAME = "cad_label"


# Return True if a single ICD code represents CAD. For bulk ops use cad_code_mask().
def is_cad_code(icd_code, icd_version):
    if pd.isna(icd_code):
        return False

    code = str(icd_code).strip().upper()

    if icd_version == 10:
        return code.startswith(("I20", "I21", "I22", "I23", "I24", "I25"))

    if icd_version == 9:
        return code.startswith(("410", "411", "412", "413", "414"))

    return False


# Vectorised boolean mask for CAD codes; used for labelling and leakage removal.
def cad_code_mask(df, code_col="icd_code", version_col="icd_version"):
    code = df[code_col].astype(str).str.strip().str.upper()
    version = df[version_col]

    mask_10 = (version == 10) & code.str.startswith(("I20", "I21", "I22", "I23", "I24", "I25"))
    mask_9 = (version == 9) & code.str.startswith(("410", "411", "412", "413", "414"))

    return mask_10 | mask_9


# Each group maps a feature name to ICD-9 and ICD-10 prefixes; produces a binary flag per admission
DX_GROUPS = {
    "dx_diabetes": {
        9: ("250",),
        10: ("E08", "E09", "E10", "E11", "E13"),
    },
    "dx_hypertension": {
        9: ("401", "402", "403", "404", "405"),
        10: ("I10", "I11", "I12", "I13", "I15"),
    },
    "dx_ckd": {
        9: ("585",),
        10: ("N18",),
    },
    "dx_hyperlipidemia": {
        9: ("272",),
        10: ("E78",),
    },
    "dx_obesity": {
        9: ("2780", "27800", "27801"),
        10: ("E66",),
    },
    "dx_heart_failure": {
        9: ("428",),
        10: ("I50",),
    },
    "dx_stroke": {
        9: ("430", "431", "432", "433", "434", "435", "436"),
        10: ("I60", "I61", "I62", "I63", "I64", "G45"),
    },
}

PROC_GROUPS = {
    "proc_cardiac_catheterization": {
        9: ("3721", "3722", "3723"),
        10: ("B21", "B24"),
    },
    "proc_coronary_angioplasty": {
        9: ("0066", "3601", "3602", "3605"),
        10: ("027",),
    },
    "proc_coronary_bypass": {
        9: ("3610", "3611", "3612", "3613", "3614"),
        10: ("0210", "0211", "0212", "0213"),
    },
}


# Load the four core MIMIC-IV tables, selecting only the columns we need.
def load_tables():
    patients = pd.read_csv(
        PATIENTS_PATH,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]
    )

    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "admission_location",
            "discharge_location", "insurance", "language",
            "marital_status", "race"
        ],
        parse_dates=["admittime", "dischtime"]
    )

    diagnoses = pd.read_csv(
        DIAGNOSES_PATH,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    )

    procedures = pd.read_csv(
        PROCEDURES_PATH,
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    )

    return patients, admissions, diagnoses, procedures


# Join admissions with patient demographics and derive age at admission.
def build_base_admission_cohort(patients, admissions):
    df = admissions.merge(patients, on="subject_id", how="left")

    df["admit_year"] = df["admittime"].dt.year
    # MIMIC-IV stores age relative to anchor_year; shift to actual admission year
    df["age_at_admit"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])

    df = df.drop_duplicates(subset=["hadm_id"]).copy()
    return df


# Label an admission 1 if any diagnosis code is CAD, 0 otherwise.
def build_cad_label(diagnoses):
    dx = diagnoses.copy()
    dx[LABEL_NAME] = cad_code_mask(dx).astype(int)

    # Max over all codes per admission: any positive code → label = 1
    label_df = (
        dx.groupby("hadm_id", as_index=False)[LABEL_NAME]
        .max()
    )
    return label_df


# Return 1 if code starts with any prefix defined for its ICD version, else 0.
def match_prefix(code, version, prefix_map):
    if pd.isna(code):
        return 0
    code = str(code).strip().upper()
    prefixes = prefix_map.get(version, ())
    return int(any(code.startswith(p) for p in prefixes))


# Create one binary flag per clinical group (DX_GROUPS or PROC_GROUPS) per admission.
def build_group_flags(df_codes, groups):
    out = df_codes[["hadm_id"]].drop_duplicates().copy()

    for feat_name, prefix_map in groups.items():
        temp = df_codes.copy()
        temp[feat_name] = temp.apply(
            lambda row: match_prefix(row["icd_code"], row["icd_version"], prefix_map),
            axis=1
        )
        feat_df = temp.groupby("hadm_id", as_index=False)[feat_name].max()
        out = out.merge(feat_df, on="hadm_id", how="left")

    for c in out.columns:
        if c != "hadm_id":
            out[c] = out[c].fillna(0).astype(int)

    return out


# One-hot encode categorical demographic/admission columns; NaNs become 'UNKNOWN'.
def one_hot_encode_base_features(df):
    df = df.copy()

    categorical_cols = [
        "gender", "admission_type", "admission_location",
        "discharge_location", "insurance", "language",
        "marital_status", "race"
    ]

    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    keep_raw_cols = [
        "subject_id", "hadm_id", "admit_year", "age_at_admit",
        "anchor_year", "anchor_year_group"
    ]
    keep_raw_cols = [c for c in keep_raw_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    out = pd.get_dummies(
        df[keep_raw_cols + categorical_cols],
        columns=categorical_cols,
        drop_first=False  # retain all levels; no reference category dropped
    )

    return out


# Pivot the top-k most frequent ICD codes into binary admission-level features.
# exclude_cad strips CAD codes before computing frequencies to prevent leakage.
def build_top_code_features(df_codes, top_k=100, feature_prefix="code", exclude_cad=False):
    temp = df_codes.copy()

    if exclude_cad:
        temp = temp.loc[~cad_code_mask(temp)].copy()

    # Combine version + code into one key, e.g. "10_I25", to avoid collisions across ICD versions
    temp["code_str"] = temp["icd_version"].astype(str) + "_" + temp["icd_code"].astype(str)

    top_codes = temp["code_str"].value_counts().head(top_k).index.tolist()
    temp = temp[temp["code_str"].isin(top_codes)].copy()

    temp["value"] = 1
    wide = (
        temp.pivot_table(
            index="hadm_id",
            columns="code_str",
            values="value",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    new_cols = []
    for c in wide.columns:
        if c == "hadm_id":
            new_cols.append(c)
        else:
            new_cols.append(f"{feature_prefix}_{c}")
    wide.columns = new_cols

    return wide


# Drop CAD codes from the diagnoses table so they cannot appear as input features.
def remove_cad_diagnoses_from_features(diagnoses):
    return diagnoses.loc[~cad_code_mask(diagnoses)].copy()


# Build three feature-set variants for temporal domain-generalisation experiments.
# Dataset A: demographics + non-CAD diagnoses
# Dataset B: demographics + non-CAD diagnoses + procedures
# Dataset C: demographics + procedures only
# CAD codes are excluded from all feature tables to prevent leakage but are
# used in full to construct the binary label.
def build_all_three_datasets():
    patients, admissions, diagnoses, procedures = load_tables()

    base = build_base_admission_cohort(patients, admissions)
    base_encoded = one_hot_encode_base_features(base)

    # Label uses the full diagnoses table; CAD codes are removed only from features below
    label_df = build_cad_label(diagnoses)

    master = base_encoded.merge(label_df, on="hadm_id", how="left")
    master[LABEL_NAME] = master[LABEL_NAME].fillna(0).astype(int)

    diagnoses_nocad = remove_cad_diagnoses_from_features(diagnoses)

    dx_group_flags = build_group_flags(diagnoses_nocad, DX_GROUPS)
    dx_top_codes = build_top_code_features(
        diagnoses_nocad,
        top_k=100,
        feature_prefix="dxcode",
        exclude_cad=False  # CAD already removed; no second filter needed
    )

    proc_group_flags = build_group_flags(procedures, PROC_GROUPS)
    proc_top_codes = build_top_code_features(
        procedures,
        top_k=50,
        feature_prefix="proccode",
        exclude_cad=False
    )

    dataset_a = (
        master
        .merge(dx_group_flags, on="hadm_id", how="left")
        .merge(dx_top_codes, on="hadm_id", how="left")
    )

    dataset_b = (
        master
        .merge(dx_group_flags, on="hadm_id", how="left")
        .merge(dx_top_codes, on="hadm_id", how="left")
        .merge(proc_group_flags, on="hadm_id", how="left")
        .merge(proc_top_codes, on="hadm_id", how="left")
    )

    dataset_c = (
        master
        .merge(proc_group_flags, on="hadm_id", how="left")
        .merge(proc_top_codes, on="hadm_id", how="left")
    )

    protected_cols = {
        "subject_id", "hadm_id", "admit_year", "age_at_admit",
        "anchor_year", "anchor_year_group", LABEL_NAME
    }

    # Fill NaNs introduced by left-merges; binary features default to 0 (absent)
    for df in [dataset_a, dataset_b, dataset_c]:
        for c in df.columns:
            if c in protected_cols:
                continue
            if df[c].dtype == object:
                df[c] = df[c].fillna("UNKNOWN")
            else:
                df[c] = df[c].fillna(0)

    return dataset_a, dataset_b, dataset_c


def summarize_dataset(df, name):
    print(f"\n{name}")
    print("-" * len(name))
    print("shape:", df.shape)
    print("unique hadm_id:", df["hadm_id"].nunique())
    print("label prevalence:", round(df[LABEL_NAME].mean(), 4))

    feature_cols = [c for c in df.columns if c not in ["subject_id", "hadm_id", LABEL_NAME]]
    print("n features:", len(feature_cols))

    if "anchor_year_group" in df.columns:
        print("\nanchor_year_group counts:")
        print(df["anchor_year_group"].value_counts(dropna=False).sort_index())


def save_datasets(dataset_a, dataset_b, dataset_c, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_a.to_csv(out_dir / "dataset_a_demo_dx_nocad.csv", index=False)
    dataset_b.to_csv(out_dir / "dataset_b_demo_dx_proc_nocad.csv", index=False)
    dataset_c.to_csv(out_dir / "dataset_c_demo_proc.csv", index=False)


if __name__ == "__main__":
    dataset_a, dataset_b, dataset_c = build_all_three_datasets()

    summarize_dataset(dataset_a, "Dataset A: demographics + non-CAD diagnoses")
    summarize_dataset(dataset_b, "Dataset B: demographics + non-CAD diagnoses + procedures")
    summarize_dataset(dataset_c, "Dataset C: demographics + procedures")

    save_datasets(dataset_a, dataset_b, dataset_c, OUT_DIR)

    print("\nSaved datasets to:")
    print(OUT_DIR)