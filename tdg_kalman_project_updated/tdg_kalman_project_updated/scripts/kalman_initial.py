import argparse
from pathlib import Path
import copy
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from pykalman import KalmanFilter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


LABEL_COL = "cad_label"
ID_COLS = ["subject_id", "hadm_id"]
TIME_CANDIDATES = ["anchor_year_group", "anchor_year"]


def normalize_anchor_year_group(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s*-\s*", "-", regex=True)
    s = s.replace({"nan": np.nan})
    return s


def sort_time_blocks(blocks):
    def key_fn(x):
        x = str(x)
        if "-" in x:
            return int(x.split("-")[0])
        return int(x)
    return sorted(blocks, key=key_fn)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing {LABEL_COL}")

    if "anchor_year_group" in df.columns:
        df["anchor_year_group"] = normalize_anchor_year_group(df["anchor_year_group"])
        df["time_block"] = df["anchor_year_group"]
    elif "anchor_year" in df.columns:
        df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
        df["time_block"] = df["anchor_year"].astype("Int64").astype(str)
    else:
        raise ValueError("Dataset must contain anchor_year_group or anchor_year")

    feature_cols = [
        c for c in df.columns
        if c not in ID_COLS + [LABEL_COL, "admit_year", "anchor_year", "anchor_year_group", "time_block"]
    ]

    blocks = sort_time_blocks(df["time_block"].dropna().unique().tolist())
    return df, feature_cols, blocks


def build_preprocessor(df, feature_cols, fit_blocks):
    fit_mask = df["time_block"].isin(fit_blocks)
    X_fit = df.loc[fit_mask, feature_cols]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_fit)
    scaler.fit(X_imp)

    return imputer, scaler


def transform_features(df_slice, feature_cols, imputer, scaler):
    X = df_slice[feature_cols]
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X.astype(np.float32)


def stratified_anchor_indices(y, anchor_size, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = min(len(pos_idx), anchor_size // 2)
    n_neg = min(len(neg_idx), anchor_size - n_pos)

    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)

    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return idx



class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_loader(X, y, batch_size=512, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict_logits(model, X, device="cpu", batch_size=2048):
    model.eval()
    outs = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size,
                        shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb).detach().cpu().numpy()
            outs.append(logits)
    return np.concatenate(outs)


def predict_probs(model, X, device="cpu", batch_size=2048):
    logits = predict_logits(model, X, device=device, batch_size=batch_size)
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_model(model, X, y, device="cpu"):
    probs = predict_probs(model, X, device=device)
    return {
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
    }


def train_block_model(
    X_train,
    y_train,
    input_dim,
    hidden_dim=128,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    epochs=20,
    device="cpu",
):
    model = TabularMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def kalman_guided_finetune(
    base_model,
    X_train,
    y_train,
    X_anchor,
    target_anchor_probs,
    lambda_reg=1.0,
    lr=5e-4,
    weight_decay=1e-5,
    batch_size=512,
    epochs=8,
    device="cpu",
):
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    X_anchor_t = torch.tensor(X_anchor, dtype=torch.float32).to(device)
    target_anchor_t = torch.tensor(target_anchor_probs, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss_ce = ce_loss(logits, yb)

            anchor_logits = model(X_anchor_t)
            anchor_probs = torch.sigmoid(anchor_logits)
            loss_reg = mse_loss(anchor_probs, target_anchor_t)

            loss = loss_ce + lambda_reg * loss_reg
            loss.backward()
            optimizer.step()

    return model


def build_observation_sequence(models, X_anchor, device="cpu"):
    obs = []
    for model in models:
        probs = predict_probs(model, X_anchor, device=device)
        obs.append(probs)
    return np.stack(obs, axis=0)  # [T, N_anchor]


def fit_kalman_on_observations(obs_matrix, max_latent_dim=5, em_iters=20, random_state=42):
    """
    obs_matrix: [T, N_anchor]
    We first PCA-compress the anchor-output observations, then fit KF on low-dim sequence.
    """
    T, obs_dim = obs_matrix.shape
    pca_dim = max(1, min(max_latent_dim, T, obs_dim))

    pca = PCA(n_components=pca_dim, random_state=random_state)
    obs_low = pca.fit_transform(obs_matrix)

    kf = KalmanFilter(
        n_dim_obs=obs_low.shape[1],
        n_dim_state=pca_dim,
        random_state=random_state
    )

    kf = kf.em(obs_low, n_iter=em_iters)
    state_means, state_covs = kf.filter(obs_low)

    next_state_mean, next_state_cov = kf.filter_update(
        filtered_state_mean=state_means[-1],
        filtered_state_covariance=state_covs[-1],
        observation=None
    )

    next_obs_low = (
        np.asarray(kf.observation_matrices) @ next_state_mean
        + np.asarray(kf.observation_offsets)
    )

    next_obs_full = pca.inverse_transform(next_obs_low.reshape(1, -1)).ravel()
    next_obs_full = np.clip(next_obs_full, 1e-5, 1 - 1e-5)

    return {
        "pca": pca,
        "kf": kf,
        "obs_low": obs_low,
        "state_means": state_means,
        "state_covs": state_covs,
        "pred_anchor_probs": next_obs_full,
    }


def run_pairwise_kf_experiment(
    df,
    feature_cols,
    blocks,
    out_dir,
    anchor_size=4000,
    hidden_dim=128,
    base_epochs=20,
    finetune_epochs=8,
    lambda_reg=1.0,
    device="cpu",
    seed=42,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # fixed anchor set from earliest block only
    first_block = blocks[0]
    first_df = df.loc[df["time_block"] == first_block].reset_index(drop=True)
    first_y = first_df[LABEL_COL].values
    anchor_idx = stratified_anchor_indices(first_y, anchor_size=anchor_size, seed=seed)
    anchor_df = first_df.iloc[anchor_idx].reset_index(drop=True)

    print(f"Using fixed anchor set from block {first_block}, size={len(anchor_df)}")

    for i in range(len(blocks) - 1):
        history_blocks = blocks[:i+1]
        current_block = blocks[i]
        next_block = blocks[i+1]

        print(f"\n=== Pair: train through {current_block} -> test on {next_block} ===")

        # Preprocessor fitted only on observed history
        imputer, scaler = build_preprocessor(df, feature_cols, history_blocks)

        X_anchor = transform_features(anchor_df, feature_cols, imputer, scaler)

        # train one model per observed block up to current
        history_models = []
        for b in history_blocks:
            block_df = df.loc[df["time_block"] == b]
            X_b = transform_features(block_df, feature_cols, imputer, scaler)
            y_b = block_df[LABEL_COL].values.astype(np.float32)

            model_b = train_block_model(
                X_train=X_b,
                y_train=y_b,
                input_dim=X_b.shape[1],
                hidden_dim=hidden_dim,
                epochs=base_epochs,
                device=device,
            )
            history_models.append(model_b)

        # build observation sequence from history models on fixed anchor set
        obs_matrix = build_observation_sequence(history_models, X_anchor, device=device)

        # KF forecast of next anchor outputs
        kf_result = fit_kalman_on_observations(obs_matrix)
        pred_anchor_probs = kf_result["pred_anchor_probs"]

        # baseline = current block model
        baseline_model = history_models[-1]

        # fine-tune current model toward predicted future anchor outputs
        current_df = df.loc[df["time_block"] == current_block]
        X_current = transform_features(current_df, feature_cols, imputer, scaler)
        y_current = current_df[LABEL_COL].values.astype(np.float32)

        adapted_model = kalman_guided_finetune(
            base_model=baseline_model,
            X_train=X_current,
            y_train=y_current,
            X_anchor=X_anchor,
            target_anchor_probs=pred_anchor_probs.astype(np.float32),
            lambda_reg=lambda_reg,
            epochs=finetune_epochs,
            device=device,
        )

        # evaluate on next block
        next_df = df.loc[df["time_block"] == next_block]
        X_next = transform_features(next_df, feature_cols, imputer, scaler)
        y_next = next_df[LABEL_COL].values.astype(np.float32)

        baseline_metrics = evaluate_model(baseline_model, X_next, y_next, device=device)
        adapted_metrics = evaluate_model(adapted_model, X_next, y_next, device=device)

        row = {
            "history_end_block": current_block,
            "test_block": next_block,
            "baseline_auroc": baseline_metrics["auroc"],
            "baseline_auprc": baseline_metrics["auprc"],
            "baseline_brier": baseline_metrics["brier"],
            "kf_auroc": adapted_metrics["auroc"],
            "kf_auprc": adapted_metrics["auprc"],
            "kf_brier": adapted_metrics["brier"],
            "delta_auroc": adapted_metrics["auroc"] - baseline_metrics["auroc"],
            "delta_auprc": adapted_metrics["auprc"] - baseline_metrics["auprc"],
            "delta_brier": adapted_metrics["brier"] - baseline_metrics["brier"],
            "n_test": len(next_df),
        }
        all_rows.append(row)

        # save per-pair anchor forecast diagnostics
        pair_df = pd.DataFrame({
            "anchor_baseline_probs": predict_probs(baseline_model, X_anchor, device=device),
            "anchor_kf_target_probs": pred_anchor_probs,
            "anchor_adapted_probs": predict_probs(adapted_model, X_anchor, device=device),
        })
        pair_df.to_csv(out_dir / f"anchor_forecast_{current_block}_to_{next_block}.csv", index=False)

        # cleanup
        del history_models, baseline_model, adapted_model
        del X_current, y_current, X_next, y_next, X_anchor, obs_matrix, kf_result
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "kf_pairwise_results.csv", index=False)
    return results_df


def plot_pairwise_comparison(results_df, metric, out_path):
    base_col = f"baseline_{metric}"
    kf_col = f"kf_{metric}"

    x = np.arange(len(results_df))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, results_df[base_col], width, label="Baseline")
    plt.bar(x + width/2, results_df[kf_col], width, label="Kalman-adapted")

    labels = [f"{a}\N{RIGHTWARDS ARROW}{b}" for a, b in zip(results_df["history_end_block"], results_df["test_block"])]
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(metric.upper())
    plt.title(f"Baseline vs Kalman-adapted ({metric.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_gain(results_df, metric, out_path):
    delta_col = f"delta_{metric}"
    x = np.arange(len(results_df))

    plt.figure(figsize=(10, 4))
    plt.axhline(0, linewidth=1)
    plt.bar(x, results_df[delta_col])

    labels = [f"{a}\N{RIGHTWARDS ARROW}{b}" for a, b in zip(results_df["history_end_block"], results_df["test_block"])]
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(f"Δ {metric.upper()}")
    plt.title(f"Kalman gain over baseline: {metric.upper()}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--anchor_size", type=int, default=4000)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--base_epochs", type=int, default=20)
    parser.add_argument("--finetune_epochs", type=int, default=8)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols, blocks = load_dataset(args.dataset_path)

    print("Loaded dataset")
    print("Shape:", df.shape)
    print("Num features:", len(feature_cols))
    print("Blocks:", blocks)

    results_df = run_pairwise_kf_experiment(
        df=df,
        feature_cols=feature_cols,
        blocks=blocks,
        out_dir=out_dir,
        anchor_size=args.anchor_size,
        hidden_dim=args.hidden_dim,
        base_epochs=args.base_epochs,
        finetune_epochs=args.finetune_epochs,
        lambda_reg=args.lambda_reg,
        device=args.device,
        seed=args.seed,
    )

    print("\nResults:")
    print(results_df)

    plot_pairwise_comparison(results_df, "auroc", out_dir / "pairwise_auroc.png")
    plot_pairwise_comparison(results_df, "auprc", out_dir / "pairwise_auprc.png")
    plot_pairwise_comparison(results_df, "brier", out_dir / "pairwise_brier.png")

    plot_metric_gain(results_df, "auroc", out_dir / "delta_auroc.png")
    plot_metric_gain(results_df, "auprc", out_dir / "delta_auprc.png")
    plot_metric_gain(results_df, "brier", out_dir / "delta_brier.png")

    summary = {
        "mean_delta_auroc": results_df["delta_auroc"].mean(),
        "mean_delta_auprc": results_df["delta_auprc"].mean(),
        "mean_delta_brier": results_df["delta_brier"].mean(),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)

    print("\nSaved outputs to:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
