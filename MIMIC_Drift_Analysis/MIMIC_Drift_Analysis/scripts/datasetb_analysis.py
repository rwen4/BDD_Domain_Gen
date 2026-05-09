import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp


LABEL_COL = "cad_label"
ID_COLS = ["subject_id", "hadm_id"]


def load_dataset(path):
    df = pd.read_csv(path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' in dataset.")

    if "anchor_year_group" not in df.columns and "anchor_year" not in df.columns:
        raise ValueError(
            "Dataset must contain 'anchor_year_group' or 'anchor_year'."
        )

    return df


# Normalise anchor_year_group strings to a consistent "YYYY-YYYY" format.
def normalize_anchor_year_group(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*-\s*", "-", regex=True)
    s = s.replace({"nan": np.nan})
    return s


# Return the dataframe with a unified "time_block" column and a sorted list of block labels.
# Prefers anchor_year_group if present; falls back to individual anchor_year values.
def get_time_blocks(df):
    df = df.copy()

    if "anchor_year_group" in df.columns:
        df["anchor_year_group"] = normalize_anchor_year_group(df["anchor_year_group"])
        blocks = sorted(
            df["anchor_year_group"].dropna().unique().tolist(),
            key=lambda x: int(str(x).split("-")[0])
        )
        df["time_block"] = df["anchor_year_group"]
        return df, blocks

    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
    df["time_block"] = df["anchor_year"].astype("Int64").astype(str)
    blocks = sorted(df["anchor_year"].dropna().astype(int).unique().tolist())
    blocks = [str(x) for x in blocks]
    return df, blocks


# Return columns to use as model inputs, excluding IDs, label, and time metadata.
def get_feature_cols(df):
    exclude = set(ID_COLS + [
        LABEL_COL, "admit_year", "anchor_year", "anchor_year_group", "time_block"
    ])
    return [c for c in df.columns if c not in exclude]


# Downcast numeric columns in-place to reduce memory usage.
def downcast_numeric(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = set()

    for c in df.columns:
        if c in exclude_cols:
            continue

        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")

    return df


# Impute → scale → logistic regression pipeline with balanced class weights.
def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])


# Train on each block, evaluate on every other block; returns a long-form results DataFrame.
def evaluate_temporal_blocks(df, feature_cols, blocks):
    rows = []
    model = make_model()

    for train_block in blocks:
        train_mask = df["time_block"] == train_block
        y_train = df.loc[train_mask, LABEL_COL]

        # Skip blocks with no samples or only one class
        if len(y_train) == 0 or y_train.nunique() < 2:
            continue

        X_train = df.loc[train_mask, feature_cols]

        fitted = clone(model)
        fitted.fit(X_train, y_train)

        for test_block in blocks:
            test_mask = df["time_block"] == test_block
            y_test = df.loc[test_mask, LABEL_COL]

            if len(y_test) == 0 or y_test.nunique() < 2:
                continue

            X_test = df.loc[test_mask, feature_cols]
            y_prob = fitted.predict_proba(X_test)[:, 1]

            rows.append({
                "train_block": str(train_block),
                "test_block": str(test_block),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "prevalence": float(y_test.mean()),
                "auroc": float(roc_auc_score(y_test, y_prob)),
                "auprc": float(average_precision_score(y_test, y_prob)),
            })

        del fitted, X_train, y_train
        gc.collect()

    return pd.DataFrame(rows)


# Train once on the first block, then evaluate on all blocks to measure temporal decay.
def forward_generalization(df, feature_cols, blocks):
    model = make_model()
    first_block = blocks[0]

    train_mask = df["time_block"] == first_block
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, LABEL_COL]

    fitted = clone(model)
    fitted.fit(X_train, y_train)

    rows = []
    for block in blocks:
        test_mask = df["time_block"] == block
        y_test = df.loc[test_mask, LABEL_COL]

        if len(y_test) == 0 or y_test.nunique() < 2:
            continue

        X_test = df.loc[test_mask, feature_cols]
        y_prob = fitted.predict_proba(X_test)[:, 1]

        rows.append({
            "test_block": str(block),
            "auroc": float(roc_auc_score(y_test, y_prob)),
            "auprc": float(average_precision_score(y_test, y_prob)),
            "n_test": int(test_mask.sum())
        })

    del fitted, X_train, y_train
    gc.collect()

    return pd.DataFrame(rows)


def plot_metric_heatmap(results_df, metric, out_path):
    pivot = results_df.pivot(index="train_block", columns="test_block", values=metric)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)

    # Annotate each cell with its numeric value
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    plt.xlabel("Test block")
    plt.ylabel("Train block")
    plt.title(f"{metric.upper()} across temporal train/test blocks")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_forward_curve(forward_df, metric, out_path):
    plt.figure(figsize=(9, 5))
    plt.plot(forward_df["test_block"], forward_df[metric], marker="o")
    plt.xlabel("Test block")
    plt.ylabel(metric.upper())
    plt.title(f"Forward temporal generalization ({metric.upper()})")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_prevalence_by_block(df, out_path):
    prev = df.groupby("time_block")[LABEL_COL].mean().reset_index()

    plt.figure(figsize=(9, 5))
    plt.plot(prev["time_block"], prev[LABEL_COL], marker="o")
    plt.xlabel("Time block")
    plt.ylabel("CAD prevalence")
    plt.title("CAD prevalence across time")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return prev


# Return True if a series contains only 0/1 values (binary feature).
def is_binary(series):
    vals = pd.Series(series).dropna().unique()
    if len(vals) == 0:
        return False
    return set(np.unique(vals)).issubset({0, 1})


# Compare earliest vs latest block per feature.
# Binary features use absolute prevalence difference; continuous features use KS statistic.
def compute_drift_scores(df, feature_cols, sample_n=100000):
    if len(df) > sample_n:
        df_work = df.sample(sample_n, random_state=42)
    else:
        df_work = df

    blocks = sorted(
        df_work["time_block"].dropna().unique().tolist(),
        key=lambda x: int(str(x).split("-")[0]) if "-" in str(x) else int(str(x))
    )

    first_block = blocks[0]
    last_block = blocks[-1]

    early = df_work.loc[df_work["time_block"] == first_block, feature_cols]
    late = df_work.loc[df_work["time_block"] == last_block, feature_cols]

    rows = []
    for col in feature_cols:
        a = early[col]
        b = late[col]

        if is_binary(df_work[col]):
            score = abs(a.mean() - b.mean())
            method = "abs prevalence diff"
        else:
            a2 = a.dropna()
            b2 = b.dropna()
            # Skip KS test if either group is too small to be reliable
            if len(a2) < 20 or len(b2) < 20:
                score = np.nan
            else:
                score = ks_2samp(a2, b2).statistic
            method = "KS statistic"

        rows.append({
            "feature": col,
            "drift_score": score,
            "method": method,
            "missing_rate": float(df_work[col].isna().mean())
        })

    out = pd.DataFrame(rows).sort_values("drift_score", ascending=False)
    return out


def plot_top_drift(drift_df, out_path, top_n=20):
    plot_df = drift_df.dropna(subset=["drift_score"]).head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(plot_df["feature"], plot_df["drift_score"])
    plt.xlabel("Drift score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} drifting features (earliest vs latest block)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# Compute per-feature missing rates for each time block.
def block_missingness(df, feature_cols, sample_n=100000):
    if len(df) > sample_n:
        df_work = df.sample(sample_n, random_state=42)
    else:
        df_work = df

    out = (
        df_work.groupby("time_block")[feature_cols]
        .apply(lambda x: x.isna().mean())
        .T
    )
    out.columns.name = None
    return out


# Show only the top_n features by missingness variance across blocks.
def plot_missingness_heatmap(missing_df, out_path, top_n=40):
    variability = missing_df.std(axis=1).sort_values(ascending=False)
    top_feats = variability.head(top_n).index
    plot_df = missing_df.loc[top_feats]

    plt.figure(figsize=(10, max(6, top_n * 0.18)))
    im = plt.imshow(plot_df.values, aspect="auto")
    plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(plot_df.index)), plot_df.index)
    plt.xlabel("Time block")
    plt.ylabel("Feature")
    plt.title(f"Missingness heatmap (top {top_n} most variable features)")
    plt.colorbar(im, label="Missing rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# Summarize performance drop: mean same-block AUROC vs mean AUROC when training on the first block only.
def summarize_future_drop(results_df):
    diag_mask = results_df["train_block"] == results_df["test_block"]
    in_domain = results_df.loc[diag_mask, "auroc"].mean()

    first_train = sorted(
        results_df["train_block"].unique(),
        key=lambda x: int(str(x).split("-")[0]) if "-" in str(x) else int(str(x))
    )[0]
    first_rows = results_df.loc[results_df["train_block"] == first_train]
    future = first_rows["auroc"].mean()

    return {
        "mean_same_block_auroc": float(in_domain),
        "mean_first_train_across_tests_auroc": float(future),
        "auroc_drop": float(in_domain - future)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset B CSV")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save figures/results")
    parser.add_argument("--drift_sample_n", type=int, default=100000)
    parser.add_argument("--missing_sample_n", type=int, default=100000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args.dataset_path)
    df, blocks = get_time_blocks(df)

    print("Downcasting numeric columns...")
    df = downcast_numeric(df, exclude_cols={"anchor_year_group", "time_block"})

    feature_cols = get_feature_cols(df)

    print(f"Shape: {df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Time blocks: {blocks}")

    print("Evaluating temporal train/test blocks...")
    results_df = evaluate_temporal_blocks(df, feature_cols, blocks)
    results_df.to_csv(out_dir / "temporal_block_results.csv", index=False)
    plot_metric_heatmap(results_df, "auroc", out_dir / "heatmap_auroc.png")
    plot_metric_heatmap(results_df, "auprc", out_dir / "heatmap_auprc.png")

    print("Running forward generalization...")
    forward_df = forward_generalization(df, feature_cols, blocks)
    forward_df.to_csv(out_dir / "forward_generalization.csv", index=False)
    plot_forward_curve(forward_df, "auroc", out_dir / "forward_auroc.png")
    plot_forward_curve(forward_df, "auprc", out_dir / "forward_auprc.png")

    print("Plotting prevalence...")
    prev_df = plot_prevalence_by_block(df, out_dir / "cad_prevalence_by_block.png")
    prev_df.to_csv(out_dir / "cad_prevalence_by_block.csv", index=False)

    print("Computing drift scores...")
    drift_df = compute_drift_scores(df, feature_cols, sample_n=args.drift_sample_n)
    drift_df.to_csv(out_dir / "drift_scores.csv", index=False)
    plot_top_drift(drift_df, out_dir / "top_drifting_features.png")

    print("Computing missingness...")
    miss_df = block_missingness(df, feature_cols, sample_n=args.missing_sample_n)
    miss_df.to_csv(out_dir / "missingness_by_block.csv")
    plot_missingness_heatmap(miss_df, out_dir / "missingness_heatmap.png")

    summary = summarize_future_drop(results_df)
    pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)

    print("Done.")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()