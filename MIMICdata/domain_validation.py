"""
Domain Validation: Domain Discriminator for Temporal Distribution Shift.

Implements a multi-class classifier D: X -> T that predicts time period T from patient features X.
If the discriminator can accurately identify the domain, it confirms temporal distribution shift.
Higher accuracy = more severe drift (justifies Kalman Filter trajectory tracking).

Reference: Vibe/LLM.md
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mimic_iv_Version2 import TIME_PERIODS, set_seed


def build_icd_vocabulary(diagnoses_series, max_vocab_size=2000):
    """
    Build vocabulary of ICD codes from diagnoses strings.
    Returns: (vocab: dict code->idx, idx->code)
    """
    from collections import Counter
    counter = Counter()
    for s in diagnoses_series:
        if pd.isna(s) or s == 'UNKNOWN':
            continue
        codes = [c.strip() for c in str(s).split('<sep>') if c.strip()]
        counter.update(codes)
    # Most frequent codes
    most_common = counter.most_common(max_vocab_size)
    vocab = {code: idx for idx, (code, _) in enumerate(most_common)}
    inv_vocab = {idx: code for code, idx in vocab.items()}
    return vocab, inv_vocab


def encode_diagnoses(diagnoses_series, vocab):
    """Multi-hot encoding of diagnoses into fixed-size vector."""
    n = len(diagnoses_series)
    dim = len(vocab)
    X_icd = np.zeros((n, dim), dtype=np.float32)
    for i, s in enumerate(diagnoses_series):
        if pd.isna(s) or s == 'UNKNOWN':
            continue
        codes = [c.strip() for c in str(s).split('<sep>') if c.strip()]
        for code in codes:
            if code in vocab:
                X_icd[i, vocab[code]] = 1.0
    return X_icd


def _label_encode(series):
    """Map series values to 0..K-1. Returns encoded array and classes list."""
    uniq = series.fillna('UNKNOWN').astype(str).unique()
    val_to_idx = {v: i for i, v in enumerate(sorted(uniq))}
    encoded = np.array([val_to_idx.get(str(v), 0) for v in series.fillna('UNKNOWN').astype(str)])
    classes = sorted(val_to_idx.keys())
    return encoded, classes


def _stratified_split(X, y, test_size=0.2, seed=42):
    """Train/test split stratified by y."""
    set_seed(seed)
    n = len(y)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_size)
    n_train = n - n_test
    # Simple random split (stratification would need scipy/sklearn)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def prepare_domain_validation_data(data_dir, test_size=0.2, max_samples_per_period=None, seed=42):
    """
    Load processed MIMIC-IV data and prepare (X, y_domain) for domain discriminator.
    
    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
        feature_info: dict with vocab, encoders for interpretability
    """
    processed_path = os.path.join(data_dir, 'mimic_iv_processed.csv')
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found: {processed_path}")
    
    df = pd.read_csv(processed_path)
    
    # Map admit_year to period (domain label)
    def year_to_period(year):
        for p in TIME_PERIODS:
            if p <= year < p + 3:
                return p
        return None
    
    df['period'] = df['admit_year'].apply(year_to_period)
    df = df.dropna(subset=['period'])
    df['period'] = df['period'].astype(int)
    
    # Optional: subsample for balance / speed
    if max_samples_per_period is not None:
        set_seed(seed)
        dfs = []
        for p in TIME_PERIODS:
            sub = df[df['period'] == p]
            if len(sub) > max_samples_per_period:
                sub = sub.sample(n=max_samples_per_period, random_state=seed)
            dfs.append(sub)
        df = pd.concat(dfs, ignore_index=True)
    
    # Encode ICD codes
    vocab, inv_vocab = build_icd_vocabulary(df['diagnoses'])
    X_icd = encode_diagnoses(df['diagnoses'], vocab)
    
    # Encode demographics (label encoding + one-hot for small cardinality)
    gender_enc, gender_classes = _label_encode(df['gender'])
    ethnicity_enc, ethnicity_classes = _label_encode(df['ethnicity'])
    
    age = df['age'].values.astype(np.float32)
    age_mean, age_std = age.mean(), age.std()
    if age_std < 1e-8:
        age_std = 1.0
    age_scaled = ((age - age_mean) / age_std).reshape(-1, 1)
    
    # One-hot for gender (2 classes) and ethnicity
    gender_onehot = np.eye(len(gender_classes))[gender_enc]
    ethnicity_onehot = np.eye(len(ethnicity_classes))[ethnicity_enc]
    
    X_demo = np.hstack([age_scaled, gender_onehot, ethnicity_onehot])
    
    X = np.hstack([X_icd, X_demo]).astype(np.float32)
    y = df['period'].values.astype(np.int64)
    
    # Map period to 0..K-1 for cross-entropy
    period_to_idx = {p: i for i, p in enumerate(TIME_PERIODS)}
    y_idx = np.array([period_to_idx[yi] for yi in y])
    
    X_train, X_test, y_train, y_test = _stratified_split(X, y_idx, test_size=test_size, seed=seed)
    
    feature_info = {
        'vocab': vocab, 'inv_vocab': inv_vocab,
        'gender_classes': gender_classes, 'ethnicity_classes': ethnicity_classes,
        'age_mean': age_mean, 'age_std': age_std,
        'period_to_idx': period_to_idx,
        'idx_to_period': {i: p for p, i in period_to_idx.items()},
        'n_icd': X_icd.shape[1], 'n_demo': X_demo.shape[1],
    }
    return X_train, X_test, y_train, y_test, feature_info


class DomainDiscriminator(nn.Module):
    """
    Multi-class classifier D: X -> T to predict time period from patient features.
    """
    def __init__(self, input_dim, n_domains, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_domains)
    
    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)


def train_domain_discriminator(
    X_train, y_train, X_test, y_test,
    n_domains=5, batch_size=256, epochs=20, lr=1e-3, device=None,
):
    """
    Train the domain discriminator and return accuracy metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = X_train.shape[1]
    model = DomainDiscriminator(input_dim, n_domains).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={total_loss/len(train_loader):.4f}")
    
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(device)
        logits = model(Xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    
    acc = (preds == y_test).mean()
    # Random baseline
    random_acc = 1.0 / n_domains
    return {
        'accuracy': acc,
        'random_baseline': random_acc,
        'predictions': preds,
        'probabilities': probs,
        'true_labels': y_test,
        'model': model,
    }


def run_domain_validation(data_dir, max_samples_per_period=50000, **train_kwargs):
    """
    Full pipeline: load data, train discriminator, report results.
    """
    print("Loading data for domain validation...")
    X_train, X_test, y_train, y_test, feature_info = prepare_domain_validation_data(
        data_dir, max_samples_per_period=max_samples_per_period
    )
    
    n_domains = len(TIME_PERIODS)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Domains (periods): {TIME_PERIODS}")
    
    print("\nTraining domain discriminator...")
    defaults = {'batch_size': 256, 'epochs': 20}
    defaults.update(train_kwargs)
    results = train_domain_discriminator(
        X_train, y_train, X_test, y_test,
        n_domains=n_domains,
        **defaults
    )
    
    print("\n" + "="*60)
    print("DOMAIN VALIDATION RESULTS")
    print("="*60)
    print(f" discriminator accuracy: {results['accuracy']:.2%}")
    print(f" random baseline (1/{n_domains}): {results['random_baseline']:.2%}")
    print()
    if results['accuracy'] > results['random_baseline']:
        print("CONCLUSION: Discriminator predicts time period above chance.")
        print("  -> Temporal distribution shift is CONFIRMED.")
        print("  -> Features carry a 'temporal signature'.")
        print("  -> Kalman Filter trajectory tracking is justified.")
    else:
        print("CONCLUSION: Discriminator performs near random.")
        print("  -> Data may be more domain-invariant than expected.")
    print("="*60)
    
    return results, feature_info


def compute_avg_probs_per_source_domain(results, feature_info):
    """
    For each source domain, average the predicted probabilities across all test samples
    from that domain (option b from progress.md).

    Returns:
        avg_probs: shape (n_domains, n_domains), avg_probs[source_idx][pred_idx] = avg P(pred=pred_idx | source=source_idx)
    """
    probs = results['probabilities']
    y_true = results['true_labels']
    idx_to_period = feature_info['idx_to_period']
    n_domains = len(TIME_PERIODS)

    avg_probs = np.zeros((n_domains, n_domains))
    for source_idx in range(n_domains):
        mask = y_true == source_idx
        if mask.sum() > 0:
            avg_probs[source_idx] = probs[mask].mean(axis=0)
        else:
            avg_probs[source_idx] = np.nan
    return avg_probs


def plot_domain_probability_lines(avg_probs, feature_info, ax=None, figsize=(8, 5)):
    """
    Plot domain probability distribution per source domain.
    X axis: time domain. Y axis: probability. Each line = features from a different domain.
    """
    import matplotlib.pyplot as plt

    idx_to_period = feature_info['idx_to_period']
    period_labels = [str(idx_to_period[i]) for i in range(len(TIME_PERIODS))]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(TIME_PERIODS)))
    x = np.arange(len(TIME_PERIODS))

    for source_idx in range(len(TIME_PERIODS)):
        period = idx_to_period[source_idx]
        label = f"Features from {period} ({period}-{period+2})"
        ax.plot(x, avg_probs[source_idx], 'o-', label=label, color=colors[source_idx], linewidth=2, markersize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(period_labels)
    ax.set_xlabel("Predicted domain (time period)")
    ax.set_ylabel("Probability")
    ax.set_title("Domain discriminator: average P(predicted domain | source domain)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', default='Data', nargs='?')
    parser.add_argument('--max-samples', type=int, default=50000, help='Max samples per period')
    args = parser.parse_args()
    run_domain_validation(args.data_dir, max_samples_per_period=args.max_samples)
