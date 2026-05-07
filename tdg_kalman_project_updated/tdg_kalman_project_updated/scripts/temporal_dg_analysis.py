import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp


LABEL_COL = "cad_label"
ID_COLS = ["subject_id", "hadm_id"]
TIME_COL = "anchor_year_group"


def load_dataset(path):
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Expected '{LABEL_COL}' in dataset.")

    if "anchor_year_group" not in df.columns and "anchor_year" not in df.columns:
        raise ValueError(
            "Dataset must contain 'anchor_year_group' or 'anchor_year'. "
            "Rebuild datasets to preserve one of these columns."
        )
    return df


def normalize_anchor_year_group(series):
    """
    Converts strings like:
      '2008 - 2010', '2008-2010'
    into canonical '2008-2010'
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*-\s*", "-", regex=True)
    return s


def get_time_blocks(df):
    """
    Prefer anchor_year_group. If unavailable, use anchor_year as fallback.
    Returns:
      time_mode: 'group' or 'year'
      blocks: ordered list of block labels or year tuples
      df: dataframe with cleaned time columns
    """
    df = df.copy()

    if "anchor_year_group" in df.columns:
        df["anchor_year_group"] = normalize_anchor_year_group(df["anchor_year_group"])
        valid = df["anchor_year_group"].dropna().unique().tolist()

        def block_key(x):
            try:
                return int(str(x).split("-")[0])
            except:
                return 999999

        blocks = sorted(valid, key=block_key)
        return "group", blocks, df

    # fallback
    years = sorted(df["anchor_year"].dropna().astype(int).unique())
    blocks = years
    return "year", blocks, df


def add_time_block(df, time_mode, blocks, out_col="time_block"):
    df = df.copy()

    if time_mode == "group":
        df[out_col] = normalize_anchor_year_group(df["anchor_year_group"])
    else:
        df[out_col] = df["anchor_year"].astype(int).astype(str)

    return df


def get_feature_cols(df):
    exclude = set(ID_COLS + [LABEL_COL, "admit_year", "anchor_year", "anchor_year_group", "time_block"])
    return [c for c in df.columns if c not in exclude]


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])


def evaluate_temporal_blocks(df, feature_cols, blocks, time_mode, label_col=LABEL_COL):
    rows = []
    model = make_model()

    for train_block in blocks:
        if time_mode == "group":
            train_df = df.loc[df["time_block"] == train_block].copy()
        else:
            train_df = df.loc[df["anchor_year"].astype(int) == int(train_block)].copy()

        if len(train_df) == 0 or train_df[label_col].nunique() < 2:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[label_col]

        fitted = clone(model)
        fitted.fit(X_train, y_train)

        for test_block in blocks:
            if time_mode == "group":
                test_df = df.loc[df["time_block"] == test_block].copy()
            else:
                test_df = df.loc[df["anchor_year"].astype(int) == int(test_block)].copy()

            if len(test_df) == 0 or test_df[label_col].nunique() < 2:
                continue

            X_test = test_df[feature_cols]
            y_test = test_df[label_col]
            y_prob = fitted.predict_proba(X_test)[:, 1]

            rows.append({
                "train_block": str(train_block),
                "test_block": str(test_block),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "prevalence": float(y_test.mean()),
                "auroc": float(roc_auc_score(y_test, y_prob)),
                "auprc": float(average_precision_score(y_test, y_prob)),
            })

    return pd.DataFrame(rows)


def plot_metric_heatmap(results_df, metric, out_path):
    pivot = results_df.pivot(index="train_block", columns="test_block", values=metric)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)

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


def forward_generalization(df, feature_cols, blocks, time_mode, label_col=LABEL_COL):
    model = make_model()
    train_block = blocks[0]

    if time_mode == "group":
        train_df = df.loc[df["time_block"] == train_block].copy()
    else:
        train_df = df.loc[df["anchor_year"].astype(int) == int(train_block)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]

    fitted = clone(model)
    fitted.fit(X_train, y_train)

    rows = []
    for block in blocks:
        if time_mode == "group":
            test_df = df.loc[df["time_block"] == block].copy()
        else:
            test_df = df.loc[df["anchor_year"].astype(int) == int(block)].copy()

        if len(test_df) == 0 or test_df[label_col].nunique() < 2:
            continue

        y_prob = fitted.predict_proba(test_df[feature_cols])[:, 1]
        rows.append({
            "test_block": str(block),
            "auroc": float(roc_auc_score(test_df[label_col], y_prob)),
            "auprc": float(average_precision_score(test_df[label_col], y_prob)),
            "n_test": len(test_df)
        })
    return pd.DataFrame(rows)


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


def plot_prevalence_by_block(df, out_path, label_col=LABEL_COL):
    prev = df.groupby("time_block")[label_col].mean().reset_index()

    plt.figure(figsize=(9, 5))
    plt.plot(prev["time_block"], prev[label_col], marker="o")
    plt.xlabel("Time block")
    plt.ylabel("CAD prevalence")
    plt.title("CAD prevalence across time")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return prev


def plot_pca(df, feature_cols, out_path, sample_n=20000):
    tmp = df.dropna(subset=["time_block"]).copy()
    if len(tmp) > sample_n:
        tmp = tmp.sample(sample_n, random_state=42)

    X = tmp[feature_cols]
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    tmp["PC1"] = pcs[:, 0]
    tmp["PC2"] = pcs[:, 1]

    plt.figure(figsize=(9, 7))
    for block_name, group in tmp.groupby("time_block"):
        plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.5, label=block_name)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)")
    plt.title("PCA projection colored by time block")
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def is_binary(series):
    vals = pd.Series(series).dropna().unique()
    if len(vals) == 0:
        return False
    return set(np.unique(vals)).issubset({0, 1})


def compute_drift_scores(df, feature_cols):
    blocks = list(df["time_block"].dropna().unique())
    blocks_sorted = sorted(blocks, key=lambda x: int(str(x).split("-")[0]) if "-" in str(x) else int(x))

    first_block = blocks_sorted[0]
    last_block = blocks_sorted[-1]

    early = df.loc[df["time_block"] == first_block, feature_cols]
    late = df.loc[df["time_block"] == last_block, feature_cols]

    rows = []
    for col in feature_cols:
        a = early[col]
        b = late[col]

        if is_binary(df[col]):
            score = abs(a.mean() - b.mean())
            method = "abs prevalence diff"
        else:
            a2 = a.dropna()
            b2 = b.dropna()
            if len(a2) < 20 or len(b2) < 20:
                score = np.nan
            else:
                score = ks_2samp(a2, b2).statistic
            method = "KS statistic"

        rows.append({
            "feature": col,
            "drift_score": score,
            "method": method,
            "missing_rate": float(df[col].isna().mean())
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


def block_missingness(df, feature_cols):
    out = (
        df.groupby("time_block")[feature_cols]
        .apply(lambda x: x.isna().mean())
        .T
    )
    out.columns.name = None
    return out


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


def summarize_future_drop(results_df):
    diag_mask = results_df["train_block"] == results_df["test_block"]
    in_domain = results_df.loc[diag_mask, "auroc"].mean()

    first_train = sorted(
        results_df["train_block"].unique(),
        key=lambda x: int(str(x).split("-")[0]) if "-" in str(x) else int(x)
    )[0]
    first_rows = results_df.loc[results_df["train_block"] == first_train].copy()
    future = first_rows["auroc"].mean()

    return {
        "mean_same_block_auroc": float(in_domain),
        "mean_first_train_across_tests_auroc": float(future),
        "auroc_drop": float(in_domain - future)
    }


def analyze_one_dataset(dataset_path, out_root):
    name = Path(dataset_path).stem
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    time_mode, blocks, df = get_time_blocks(df)
    df = add_time_block(df, time_mode, blocks)

    feature_cols = get_feature_cols(df)

    print(f"\nAnalyzing {name}")
    print(f"Shape: {df.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Time mode: {time_mode}")
    print(f"Blocks: {blocks}")

    results_df = evaluate_temporal_blocks(df, feature_cols, blocks, time_mode)
    results_df.to_csv(out_dir / "temporal_block_results.csv", index=False)

    plot_metric_heatmap(results_df, "auroc", out_dir / "heatmap_auroc.png")
    plot_metric_heatmap(results_df, "auprc", out_dir / "heatmap_auprc.png")

    forward_df = forward_generalization(df, feature_cols, blocks, time_mode)
    forward_df.to_csv(out_dir / "forward_generalization.csv", index=False)
    plot_forward_curve(forward_df, "auroc", out_dir / "forward_auroc.png")
    plot_forward_curve(forward_df, "auprc", out_dir / "forward_auprc.png")

    prev_df = plot_prevalence_by_block(df, out_dir / "cad_prevalence_by_block.png")
    prev_df.to_csv(out_dir / "cad_prevalence_by_block.csv", index=False)

    plot_pca(df, feature_cols, out_dir / "pca_by_block.png")

    drift_df = compute_drift_scores(df, feature_cols)
    drift_df.to_csv(out_dir / "drift_scores.csv", index=False)
    plot_top_drift(drift_df, out_dir / "top_drifting_features.png")

    miss_df = block_missingness(df, feature_cols)
    miss_df.to_csv(out_dir / "missingness_by_block.csv")
    plot_missingness_heatmap(miss_df, out_dir / "missingness_heatmap.png")

    summary = summarize_future_drop(results_df)
    summary_df = pd.DataFrame([{
        "dataset": name,
        **summary
    }])
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    return summary_df


def plot_dataset_comparison(summary_df, out_path):
    plot_df = summary_df.copy()
    plot_df = plot_df.set_index("dataset")[["mean_same_block_auroc", "mean_first_train_across_tests_auroc"]]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(plot_df.index))
    width = 0.35

    plt.bar(x - width / 2, plot_df["mean_same_block_auroc"], width, label="Same-block AUROC")
    plt.bar(x + width / 2, plot_df["mean_first_train_across_tests_auroc"], width, label="First-train mean AUROC")

    plt.xticks(x, plot_df.index, rotation=15)
    plt.ylabel("AUROC")
    plt.title("Dataset-level temporal generalization comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing dataset CSV files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save figures/results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(input_dir.glob("dataset_*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(f"No files matching dataset_*.csv found in {input_dir}")

    summary_list = []
    for path in dataset_paths:
        summary_df = analyze_one_dataset(path, out_dir)
        summary_list.append(summary_df)

    all_summary = pd.concat(summary_list, ignore_index=True)
    all_summary.to_csv(out_dir / "dataset_comparison_summary.csv", index=False)
    plot_dataset_comparison(all_summary, out_dir / "dataset_comparison.png")

    print("\nDone.")
    print(f"Saved all outputs to: {out_dir}")


if __name__ == "__main__":
    main()
