import os
import csv
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from pykalman import KalmanFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# SETTINGS
OUT_DIR = "./kalman_3d_recursive_results"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

T = 40 # training domain
N_TEST = 20 # testing domain
TOTAL = T + N_TEST

N_SAMPLES = 600 # number of points in the dataset
NOISE = 0.10
DRIFT_NOISE = 0.06

MLP_EPOCHS = 250
MLP_LR = 5e-3

STATE_DIM = 6
PCA_DIM = 6
KF_EM_ITERS = 30

ADAPT_STEPS = 250
ADAPT_LR = 3e-3
ADAPT_LAMBDA = 1e-4

N_FOLDS = 3


# DOMAIN DRIFT
def chirp_theta(t):
    return np.pi * np.sin(2 * np.pi * t / TOTAL) # sinusoidal theta trajectory


def chirp_theta_noisy(t):
    rng = np.random.RandomState(RANDOM_SEED + t * 31)
    return chirp_theta(t) + rng.randn() * DRIFT_NOISE

# DATASET
def make_moons_3d(n_samples, noise, rng):
    n = n_samples // 2

    angles0 = np.linspace(0, np.pi, n)
    X0 = np.stack([
        np.cos(angles0),
        np.sin(angles0),
        np.zeros(n)
    ], axis=1)

    angles1 = np.linspace(0, np.pi, n)
    X1 = np.stack([
        1 - np.cos(angles1),
        0.5 - np.sin(angles1),
        np.zeros(n)
    ], axis=1)

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * n + [1] * n)

    X += rng.randn(*X.shape).astype(np.float32) * noise

    return X, y


def rotate_points_z(X, theta): # rotate the dataset about the z axis
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return (R @ X.T).T


def get_dataset_3d(t, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed + t * 100)

    X, y = make_moons_3d(N_SAMPLES, NOISE, rng) # create static 2 moons
    theta = chirp_theta_noisy(t) # generate rotation angle array

    X = rotate_points_z(X, theta) # rotate teh static 2 moons to the angles stored in angle array

    return X, y


# Simple Neural Network for Per-Domain Training
class MLP3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y):
    model = MLP3D()

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=MLP_LR)

    loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        ),
        batch_size=128,
        shuffle=True
    )

    model.train()

    for _ in range(MLP_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    return model


def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).argmax(1).numpy()

    return (preds == y).mean()


# 3D ANCHORS -- ring-shaped because two moon dataset is roughly circular. This method is to ensure anchor's relevance to the data. 
def make_anchor_grid_3d():
    rng = np.random.RandomState(RANDOM_SEED)

    anchors = []

    n_ring = 96
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)

    # Dense XY-plane rings
    for r in [0.25, 0.45, 0.70, 0.95, 1.20, 1.45]:
        ring = np.stack([
            r * np.cos(angles),
            r * np.sin(angles),
            np.zeros(n_ring)
        ], axis=1)
        anchors.append(ring)

    # Rings slightly above and below the XY plane
    for z in [-0.30, -0.15, 0.15, 0.30]:
        for r in [0.55, 0.95, 1.35]:
            ring = np.stack([
                r * np.cos(angles),
                r * np.sin(angles),
                np.full(n_ring, z)
            ], axis=1)
            anchors.append(ring)

    # Random boundary-region anchors
    n_mid = 300
    mid = rng.randn(n_mid, 3).astype(np.float32) * 0.55
    mid[:, 2] *= 0.30
    anchors.append(mid)

    anchors = np.vstack(anchors).astype(np.float32)

    print(f"Anchor count: {len(anchors)}")

    return anchors


def get_anchor_outputs(model, anchors):
    model.eval() # evaluate the output at the anchors

    with torch.no_grad():
        logits = model(torch.FloatTensor(anchors))
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()

    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    return np.clip(probs, 0, 1)


# KALMAN IN PCA ANCHOR SPACE
def pca_kalman_predict(obs):
    obs = np.asarray(obs, dtype=np.float64)
    obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)

    n_obs, obs_dim = obs.shape

    # Need enough observations for PCA + Kalman to be meaningful(if too little observations, we skip)
    if n_obs < 6:
        return np.clip(obs[-1], 0, 1)

    n_comp = min(PCA_DIM, n_obs - 1, obs_dim)

    if n_comp < 2:
        return np.clip(obs[-1], 0, 1)

    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(obs)

    # Kalman filter needs a fixed-size state vector — pad or truncate to match STATE_DIM
    if n_comp < STATE_DIM:
        Z_state = np.hstack([Z, np.zeros((n_obs, STATE_DIM - n_comp))])
    else:
        Z_state = Z[:, :STATE_DIM]

    try:
        kf = KalmanFilter(
            transition_matrices=np.eye(STATE_DIM),
            observation_matrices=np.eye(STATE_DIM),
            transition_covariance=np.eye(STATE_DIM) * 1e-3,
            observation_covariance=np.eye(STATE_DIM) * 5e-3,
            initial_state_mean=Z_state[0],
            initial_state_covariance=np.eye(STATE_DIM),
            # These will be overridden by EM — serve only as starting estimates
            em_vars=[
                "transition_matrices",
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance"
            ]
        )

        kf = kf.em(Z_state, n_iter=KF_EM_ITERS)
        means, covs = kf.filter(Z_state)

        # One-step-ahead prediction with no new observation
        next_state, _ = kf.filter_update(means[-1], covs[-1], observation=None)

        # Drop padding dims, then project back to original space
        z_pred = next_state[:n_comp]
        pred = pca.inverse_transform(z_pred.reshape(1, -1))[0]

        return np.clip(pred, 0, 1)

    except Exception:
        # Graceful fallback if EM diverges or matrix ops fail
        return np.clip(obs[-1], 0, 1)



# ADAPTATION

def adapt_model(base_model, anchors, target_probs):
    model = deepcopy(base_model)

    opt = optim.SGD(
        model.parameters(),
        lr=ADAPT_LR,
        momentum=0.9,
        nesterov=True
    )

    X_anchor = torch.FloatTensor(anchors)

    target = torch.FloatTensor(
        np.stack([
            1 - target_probs,
            target_probs
        ], axis=1)
    )

    old_params = [p.detach().clone() for p in model.parameters()]

    for _ in range(ADAPT_STEPS): # tune the model to return target probability at teh anchors of interest
        model.train()
        opt.zero_grad()

        probs = torch.softmax(model(X_anchor), dim=1)

        loss_anchor = torch.mean((probs - target) ** 2)

        loss_reg = sum(
            torch.sum((p - p_old) ** 2)
            for p, p_old in zip(model.parameters(), old_params)
        )

        loss = loss_anchor + ADAPT_LAMBDA * loss_reg

        loss.backward()
        opt.step()

    return model



# PLOTS
def plot_anchor_probability_evolution(anchor_history, n_anchors=20):
    anchor_history = np.asarray(anchor_history)

    mean = np.mean(anchor_history, axis=0)
    var = np.var(anchor_history, axis=0)
    smoothness = np.mean(np.abs(np.diff(anchor_history, axis=0)), axis=0)

    mask = (mean > 0.2) & (mean < 0.8)

    if np.any(mask):
        candidate_idxs = np.where(mask)[0]
    else:
        candidate_idxs = np.arange(anchor_history.shape[1])

    score = var[candidate_idxs] / (smoothness[candidate_idxs] + 1e-6)

    selected = candidate_idxs[np.argsort(score)[-n_anchors:]]

    probs = anchor_history[:, selected].mean(axis=1)
    domains = np.arange(len(probs))

    plt.figure(figsize=(12, 5))

    plt.axvspan(0, T - 0.5, alpha=0.08, label="Observed training domains")
    plt.axvline(T - 0.5, linestyle="--", linewidth=1)

    plt.plot(
        domains,
        probs,
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Mean of {n_anchors} smooth informative anchors"
    )

    plt.xlabel("Domain")
    plt.ylabel("Mean Class-1 Probability")
    plt.ylim(-0.05, 1.05)
    plt.title("Representative Anchor Probability Evolution Across Domains")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = f"{OUT_DIR}/anchor_probability_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved anchor probability plot → {out}")

def plot_dataset_examples():
    domains = np.linspace(0, TOTAL - 1, 6, dtype=int)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "3D Two-Moons Domains with Sinusoidal Rotation Drift",
        fontsize=14,
        fontweight="bold"
    )

    for i, t in enumerate(domains):
        X, y = get_dataset_3d(t)

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")

        ax.scatter(
            X[y == 0, 0],
            X[y == 0, 1],
            X[y == 0, 2],
            s=10,
            alpha=0.5
        )

        ax.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            X[y == 1, 2],
            s=10,
            alpha=0.5
        )

        split = "TRAIN" if t < T else "TEST"

        ax.set_title(
            f"Domain {t} [{split}] "
            f"theta={np.degrees(chirp_theta_noisy(t)):.1f} deg"
        )

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)

    plt.tight_layout()

    out = f"{OUT_DIR}/dataset_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved dataset plot → {out}")


def plot_results(avg_results):
    domains = np.array([r["domain"] for r in avg_results])
    vals = np.array([r["kalman"] for r in avg_results])
    stds = np.array([r["kalman_std"] for r in avg_results])

    plt.figure(figsize=(13, 6))

    plt.axvspan(0, T - 0.5, alpha=0.08, label="Observed training domains")
    plt.axvline(T - 0.5, linestyle="--", linewidth=1)

    plt.plot(
        domains,
        vals,
        marker="o",
        linewidth=2,
        label="Closed-loop Kalman-guided adaptation"
    )

    plt.fill_between(
        domains,
        vals - stds,
        vals + stds,
        alpha=0.15
    )

    plt.xlabel("Domain")
    plt.ylabel("Classification Accuracy")
    plt.ylim(0, 1.05)

    plt.title("3D Anchor-Based Closed-Loop Kalman Adaptation Accuracy")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    out = f"{OUT_DIR}/kalman_accuracy_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved accuracy plot → {out}")


# ============================================================
# EXPERIMENT
# ============================================================

def run_single_fold(fold_seed):
    anchors = make_anchor_grid_3d()

    train_models = []
    observations = []

    print("Training per-domain MLPs...")

    for t in range(T):
        X, y = get_dataset_3d(t, seed=fold_seed)

        model_t = train_mlp(X, y)
        train_models.append(model_t)

        anchor_out = get_anchor_outputs(model_t, anchors)
        observations.append(anchor_out)

        print(
            f"  Train domain {t:2d} | "
            f"anchor var={np.var(anchor_out):.4f}"
        )

    results = []

    forecast_history = [obs.copy() for obs in observations]
    base_test_model = train_models[-1]

    anchor_history_for_plot = [obs.copy() for obs in observations]

    for t in range(TOTAL):
        X, y = get_dataset_3d(t, seed=fold_seed)

        split = "TRAIN" if t < T else "TEST"

        if t == 0:
            model_kalman = train_models[0]
            acc_kalman = evaluate(model_kalman, X, y)

        elif t < T:
            # Use previous observed training classifiers only.
            obs_hist = np.array(observations[:t])
            base = train_models[t - 1]

            target_kalman = pca_kalman_predict(obs_hist)
            model_kalman = adapt_model(base, anchors, target_kalman)

            acc_kalman = evaluate(model_kalman, X, y)

        else:
            # Online closed-loop test setting:
            # predict/evaluate current domain, then observe it for future use.
            obs_hist = np.array(forecast_history)

            target_kalman = pca_kalman_predict(obs_hist)

            model_kalman = adapt_model(
                base_test_model,
                anchors,
                target_kalman
            )

            acc_kalman = evaluate(model_kalman, X, y)

            # After evaluation, current test domain becomes observed.
            # Train real model on current domain and append its real anchor output.
            model_t = train_mlp(X, y)
            real_anchor_out = get_anchor_outputs(model_t, anchors)

            forecast_history.append(real_anchor_out.copy())
            anchor_history_for_plot.append(real_anchor_out.copy())

            base_test_model = model_t

        results.append({
            "domain": t,
            "split": split,
            "kalman": acc_kalman,
            "theta": chirp_theta_noisy(t)
        })

        print(
            f"Domain {t:2d} [{split}] | "
            f"kalman={acc_kalman:.3f}"
        )

    # Only plot one representative fold to avoid overwriting many figures.
    if fold_seed == RANDOM_SEED:
        plot_anchor_probability_evolution(anchor_history_for_plot)

    return results


def run_experiment():
    print("=" * 70)
    print("3D Anchor-Based Kalman Scaling Experiment")
    print("Closed-Loop Online Version")
    print("=" * 70)

    all_results = []

    fold_seeds = [
        RANDOM_SEED + 1000 * i
        for i in range(N_FOLDS)
    ]

    for i, seed in enumerate(fold_seeds):
        print("\n" + "=" * 50)
        print(f"Fold {i + 1}/{N_FOLDS} | seed={seed}")
        print("=" * 50)

        fold_results = run_single_fold(seed)
        all_results.append(fold_results)

    return all_results


def aggregate_results(all_results):
    avg_results = []

    for t in range(TOTAL):
        rows = [fold[t] for fold in all_results]
        vals = np.array([r["kalman"] for r in rows])

        avg_results.append({
            "domain": t,
            "split": rows[0]["split"],
            "theta": np.mean([r["theta"] for r in rows]),
            "kalman": vals.mean(),
            "kalman_std": vals.std()
        })

    return avg_results


def summarize(avg_results):
    test_rows = [r for r in avg_results if r["split"] == "TEST"]
    test_vals = np.array([r["kalman"] for r in test_rows])

    print("\n" + "=" * 70)
    print("KALMAN TEST PERFORMANCE")
    print("=" * 70)
    print(f"Mean test accuracy : {test_vals.mean():.4f}")
    print(f"Std across domains : {test_vals.std():.4f}")
    print(f"Min test accuracy  : {test_vals.min():.4f}")
    print(f"Max test accuracy  : {test_vals.max():.4f}")


def save_csv(avg_results):
    out = f"{OUT_DIR}/kalman_results.csv"

    keys = list(avg_results[0].keys())

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(avg_results)

    print(f"Saved CSV → {out}")

if __name__ == "__main__":
    plot_dataset_examples()

    all_results = run_experiment()

    avg_results = aggregate_results(all_results)

    summarize(avg_results)

    plot_results(avg_results)

    save_csv(avg_results)

    print("\nDone.")

    # # # """
# # # Kalman vs Static vs Linear Extrapolation
# # # Robust Domain Drift Experiment

# # # Main fixes:
# # # 1. Kalman predicts the latent domain angle theta.
# # # 2. Prediction is made BEFORE seeing the current test domain.
# # # 3. PCA angle is calibrated against canonical orientation.
# # # 4. Kalman uses harmonic dynamics matched to sinusoidal drift.
# # # 5. Static model receives no correction.
# # # 6. Linear extrapolation is a naive temporal baseline.
# # # """

# # # import os
# # # import warnings
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim

# # # from torch.utils.data import DataLoader, TensorDataset

# # # import matplotlib
# # # matplotlib.use("Agg")
# # # import matplotlib.pyplot as plt

# # # warnings.filterwarnings("ignore")


# # # # ============================================================
# # # # SETTINGS
# # # # ============================================================

# # # RANDOM_SEED = 42

# # # np.random.seed(RANDOM_SEED)
# # # torch.manual_seed(RANDOM_SEED)

# # # OUT_DIR = "./kalman_results"
# # # os.makedirs(OUT_DIR, exist_ok=True)

# # # T = 30
# # # N_TEST = 30
# # # TOTAL = T + N_TEST

# # # N_SAMPLES = 100
# # # NOISE = 0.10
# # # DRIFT_NOISE = 0.25

# # # MLP_EPOCHS = 150
# # # MLP_LR = 5e-3


# # # # ============================================================
# # # # DRIFT
# # # # ============================================================

# # # def chirp_theta(t):
# # #     return np.pi * np.sin(2 * np.pi * t / TOTAL)


# # # def chirp_theta_noisy(t):
# # #     rng = np.random.RandomState(RANDOM_SEED + t * 13)
# # #     return chirp_theta(t) + rng.randn() * DRIFT_NOISE


# # # # ============================================================
# # # # DATASET
# # # # ============================================================

# # # def make_moons_3d(n_samples, noise, rng):
# # #     n = n_samples // 2

# # #     angles0 = np.linspace(0, np.pi, n)
# # #     X0 = np.stack([
# # #         np.cos(angles0),
# # #         np.sin(angles0),
# # #         np.zeros(n)
# # #     ], axis=1)

# # #     angles1 = np.linspace(0, np.pi, n)
# # #     X1 = np.stack([
# # #         1 - np.cos(angles1),
# # #         0.5 - np.sin(angles1),
# # #         np.zeros(n)
# # #     ], axis=1)

# # #     X = np.vstack([X0, X1]).astype(np.float32)
# # #     y = np.array([0] * n + [1] * n)

# # #     X += rng.randn(*X.shape).astype(np.float32) * noise

# # #     return X, y


# # # def rotate_points(X, theta):
# # #     R = np.array([
# # #         [np.cos(theta), -np.sin(theta), 0],
# # #         [np.sin(theta),  np.cos(theta), 0],
# # #         [0, 0, 1]
# # #     ], dtype=np.float32)

# # #     return (R @ X.T).T


# # # def rotate_back(X, theta):
# # #     return rotate_points(X, -theta)


# # # def get_dataset_3d(t, seed=RANDOM_SEED):
# # #     rng = np.random.RandomState(seed + t * 100)

# # #     theta = chirp_theta_noisy(t)

# # #     X, y = make_moons_3d(N_SAMPLES, NOISE, rng)
# # #     X = rotate_points(X, theta)

# # #     return X, y


# # # # ============================================================
# # # # ANGLE ESTIMATION
# # # # ============================================================

# # # _prev_principal = None

# # # def reset_angle_estimator():
# # #     global _prev_principal
# # #     _prev_principal = None


# # # def estimate_theta_from_data(X):
# # #     """
# # #     PCA angle estimate with sign consistency.

# # #     Important:
# # #     PCA eigenvectors have sign ambiguity:
# # #         v and -v are both valid.

# # #     The dot-product check prevents sudden 180-degree flips.
# # #     """

# # #     global _prev_principal

# # #     xy = X[:, :2]

# # #     cov = np.cov(xy.T)
# # #     eigvals, eigvecs = np.linalg.eigh(cov)

# # #     principal = eigvecs[:, np.argmax(eigvals)]

# # #     if _prev_principal is not None:
# # #         if np.dot(principal, _prev_principal) < 0:
# # #             principal = -principal

# # #     _prev_principal = principal.copy()

# # #     theta = np.arctan2(principal[1], principal[0])

# # #     return theta


# # # # ============================================================
# # # # MODEL
# # # # ============================================================

# # # class MLP3D(nn.Module):
# # #     def __init__(self):
# # #         super().__init__()

# # #         self.net = nn.Sequential(
# # #             nn.Linear(3, 32),
# # #             nn.ReLU(),

# # #             nn.Linear(32, 32),
# # #             nn.ReLU(),

# # #             nn.Linear(32, 2)
# # #         )

# # #     def forward(self, x):
# # #         return self.net(x)


# # # def train_mlp(X, y):
# # #     model = MLP3D()

# # #     loss_fn = nn.CrossEntropyLoss()
# # #     opt = optim.Adam(model.parameters(), lr=MLP_LR)

# # #     loader = DataLoader(
# # #         TensorDataset(
# # #             torch.FloatTensor(X),
# # #             torch.LongTensor(y)
# # #         ),
# # #         batch_size=64,
# # #         shuffle=True
# # #     )

# # #     model.train()

# # #     for _ in range(MLP_EPOCHS):
# # #         for xb, yb in loader:
# # #             opt.zero_grad()
# # #             loss = loss_fn(model(xb), yb)
# # #             loss.backward()
# # #             opt.step()

# # #     return model


# # # def evaluate(model, X, y):
# # #     model.eval()

# # #     with torch.no_grad():
# # #         preds = model(torch.FloatTensor(X)).argmax(1).numpy()

# # #     return (preds == y).mean()


# # # # ============================================================
# # # # BASELINES
# # # # ============================================================

# # # def linear_predict_theta(theta_obs):
# # #     """
# # #     Naive linear extrapolation.
# # #     """

# # #     t = np.arange(len(theta_obs))
# # #     coef = np.polyfit(t, theta_obs, 1)
# # #     pred = np.polyval(coef, len(theta_obs))

# # #     return pred


# # # def kalman_predict_theta(theta_obs):
# # #     """
# # #     Harmonic Kalman filter.

# # #     State:
# # #         x = [theta, angular_velocity]

# # #     Dynamics:
# # #         theta'' + omega^2 theta = 0

# # #     This matches the sinusoidal domain drift.
# # #     """

# # #     dt = 1.0
# # #     omega = 2 * np.pi / TOTAL

# # #     A = np.array([
# # #         [1, dt],
# # #         [-omega ** 2 * dt, 1]
# # #     ])

# # #     H = np.array([[1, 0]])

# # #     Q = np.array([
# # #         [5e-3, 0],
# # #         [0, 1e-3]
# # #     ])

# # #     R = np.array([[0.05]])

# # #     x = np.array([theta_obs[0], 0.0])
# # #     P = np.eye(2)

# # #     for z in theta_obs:
# # #         # Predict
# # #         x = A @ x
# # #         P = A @ P @ A.T + Q

# # #         # Update
# # #         y = np.array([z]) - H @ x
# # #         S = H @ P @ H.T + R
# # #         K = P @ H.T @ np.linalg.inv(S)

# # #         x = x + (K @ y).flatten()
# # #         P = (np.eye(2) - K @ H) @ P

# # #     # Predict next theta
# # #     x_future = A @ x

# # #     return x_future[0]


# # # # ============================================================
# # # # PLOTS
# # # # ============================================================

# # # def plot_dataset_examples():
# # #     domains = np.linspace(0, TOTAL - 1, 6, dtype=int)

# # #     fig = plt.figure(figsize=(18, 10))
# # #     fig.suptitle(
# # #         "3D Two-Moons Dataset with Sinusoidal Rotation Drift",
# # #         fontsize=14,
# # #         fontweight="bold"
# # #     )

# # #     for i, t in enumerate(domains):
# # #         X, y = get_dataset_3d(t)

# # #         theta_deg = np.degrees(chirp_theta_noisy(t))
# # #         split = "TRAIN" if t < T else "TEST"

# # #         ax = fig.add_subplot(2, 3, i + 1, projection="3d")

# # #         ax.scatter(
# # #             X[y == 0, 0],
# # #             X[y == 0, 1],
# # #             X[y == 0, 2],
# # #             s=10,
# # #             alpha=0.5,
# # #             label="Class 0"
# # #         )

# # #         ax.scatter(
# # #             X[y == 1, 0],
# # #             X[y == 1, 1],
# # #             X[y == 1, 2],
# # #             s=10,
# # #             alpha=0.5,
# # #             label="Class 1"
# # #         )

# # #         ax.set_title(
# # #             f"Domain {t} [{split}] θ={theta_deg:.1f}°",
# # #             fontsize=10
# # #         )

# # #         ax.set_xlim(-2, 2)
# # #         ax.set_ylim(-2, 2)
# # #         ax.set_zlim(-1, 1)

# # #         ax.set_xlabel("X")
# # #         ax.set_ylabel("Y")
# # #         ax.set_zlabel("Z")

# # #     plt.tight_layout()

# # #     out = f"{OUT_DIR}/dataset_examples.png"
# # #     plt.savefig(out, dpi=150, bbox_inches="tight")
# # #     plt.close()

# # #     print(f"Saved dataset examples → {out}")


# # # def plot_results(results):
# # #     domains = np.array([r["domain"] for r in results])

# # #     static = np.array([r["static"] for r in results])
# # #     linear = np.array([r["linear"] for r in results])
# # #     kalman = np.array([r["kalman"] for r in results])

# # #     theta_true = np.array([r["theta_true"] for r in results])
# # #     theta_est = np.array([r["theta_est"] for r in results])
# # #     theta_lin = np.array([r["theta_linear_pred"] for r in results])
# # #     theta_kal = np.array([r["theta_kalman_pred"] for r in results])

# # #     # Accuracy plot
# # #     plt.figure(figsize=(13, 6))

# # #     plt.axvspan(0, T - 0.5, alpha=0.08, label="Train domains")
# # #     plt.axvline(T - 0.5, linestyle="--", linewidth=1)

# # #     plt.plot(domains, static, "o-", linewidth=2, label="Static")
# # #     plt.plot(domains, linear, "s-", linewidth=2, label="Linear extrapolation")
# # #     plt.plot(domains, kalman, "^-", linewidth=2, label="Kalman")

# # #     plt.xlabel("Domain")
# # #     plt.ylabel("Accuracy")
# # #     plt.ylim(0, 1.05)

# # #     plt.title("Kalman vs Static vs Linear under Domain Drift")
# # #     plt.legend()
# # #     plt.grid(alpha=0.3)

# # #     plt.tight_layout()

# # #     out = f"{OUT_DIR}/accuracy_results.png"
# # #     plt.savefig(out, dpi=150, bbox_inches="tight")
# # #     plt.close()

# # #     print(f"Saved accuracy plot → {out}")

# # #     # Theta prediction plot
# # #     plt.figure(figsize=(13, 6))

# # #     plt.axvspan(0, T - 0.5, alpha=0.08, label="Train domains")
# # #     plt.axvline(T - 0.5, linestyle="--", linewidth=1)

# # #     plt.plot(domains, theta_true, "k-", linewidth=2, label="True noisy theta")
# # #     plt.plot(domains, theta_est, "o-", linewidth=1.5, label="Observed estimated theta")
# # #     plt.plot(domains, theta_lin, "s--", linewidth=1.5, label="Linear predicted theta")
# # #     plt.plot(domains, theta_kal, "^--", linewidth=1.5, label="Kalman predicted theta")

# # #     plt.xlabel("Domain")
# # #     plt.ylabel("Theta radians")

# # #     plt.title("Latent Rotation Angle Tracking")
# # #     plt.legend()
# # #     plt.grid(alpha=0.3)

# # #     plt.tight_layout()

# # #     out = f"{OUT_DIR}/theta_tracking.png"
# # #     plt.savefig(out, dpi=150, bbox_inches="tight")
# # #     plt.close()

# # #     print(f"Saved theta tracking plot → {out}")


# # # # ============================================================
# # # # EXPERIMENT
# # # # ============================================================

# # # def run_experiment():
# # #     print("=" * 60)
# # #     print("Kalman Robust Domain Drift Experiment")
# # #     print("=" * 60)

# # #     print(f"T={T}, N_TEST={N_TEST}, TOTAL={TOTAL}")
# # #     print(f"N_SAMPLES={N_SAMPLES}, NOISE={NOISE}, DRIFT_NOISE={DRIFT_NOISE}")

# # #     # --------------------------------------------------------
# # #     # Train canonical classifier
# # #     # --------------------------------------------------------

# # #     print("\nTraining canonical classifier...")

# # #     rng = np.random.RandomState(RANDOM_SEED)
# # #     X0, y0 = make_moons_3d(N_SAMPLES, NOISE, rng)

# # #     model = train_mlp(X0, y0)

# # #     canonical_acc = evaluate(model, X0, y0)
# # #     print(f"Canonical training accuracy: {canonical_acc:.3f}")

# # #     # --------------------------------------------------------
# # #     # Estimate reference PCA angle
# # #     # --------------------------------------------------------

# # #     reset_angle_estimator()
# # #     theta_ref = estimate_theta_from_data(X0)

# # #     print(f"Canonical PCA reference angle: {np.degrees(theta_ref):.2f}°")

# # #     # --------------------------------------------------------
# # #     # Sequential experiment
# # #     # --------------------------------------------------------

# # #     theta_history = []
# # #     results = []

# # #     reset_angle_estimator()

# # #     for t in range(TOTAL):
# # #         X, y = get_dataset_3d(t)

# # #         theta_true = chirp_theta_noisy(t)

# # #         # ----------------------------------------------------
# # #         # Predict theta_t BEFORE observing domain t
# # #         # ----------------------------------------------------

# # #         if len(theta_history) == 0:
# # #             theta_lin = 0.0
# # #             theta_kal = 0.0

# # #         elif len(theta_history) < 3:
# # #             theta_lin = theta_history[-1]
# # #             theta_kal = theta_history[-1]

# # #         else:
# # #             obs = np.unwrap(theta_history)

# # #             theta_lin = linear_predict_theta(obs)
# # #             theta_kal = kalman_predict_theta(obs)

# # #         # ----------------------------------------------------
# # #         # Evaluate methods on current domain
# # #         # ----------------------------------------------------

# # #         acc_static = evaluate(model, X, y)

# # #         X_lin = rotate_back(X, theta_lin)
# # #         X_kal = rotate_back(X, theta_kal)

# # #         acc_lin = evaluate(model, X_lin, y)
# # #         acc_kal = evaluate(model, X_kal, y)

# # #         # ----------------------------------------------------
# # #         # Now observe current theta for future prediction
# # #         # ----------------------------------------------------

# # #         theta_est_raw = estimate_theta_from_data(X)
# # #         theta_est = theta_est_raw - theta_ref

# # #         theta_history.append(theta_est)

# # #         split = "TRAIN" if t < T else "TEST"

# # #         results.append({
# # #             "domain": t,
# # #             "split": split,

# # #             "theta_true": theta_true,
# # #             "theta_est": theta_est,

# # #             "theta_linear_pred": theta_lin,
# # #             "theta_kalman_pred": theta_kal,

# # #             "static": acc_static,
# # #             "linear": acc_lin,
# # #             "kalman": acc_kal
# # #         })

# # #         print(
# # #             f"Domain {t:2d} [{split}] | "
# # #             f"static={acc_static:.3f} | "
# # #             f"linear={acc_lin:.3f} | "
# # #             f"kalman={acc_kal:.3f} | "
# # #             f"theta_true={np.degrees(theta_true):7.2f}° | "
# # #             f"theta_kal={np.degrees(theta_kal):7.2f}°"
# # #         )

# # #     return results


# # # def summarize_results(results):
# # #     print("\n" + "=" * 60)
# # #     print("TEST AVERAGES")
# # #     print("=" * 60)

# # #     test_rows = [r for r in results if r["split"] == "TEST"]

# # #     static_avg = np.mean([r["static"] for r in test_rows])
# # #     linear_avg = np.mean([r["linear"] for r in test_rows])
# # #     kalman_avg = np.mean([r["kalman"] for r in test_rows])

# # #     print(f"Static : {static_avg:.4f}")
# # #     print(f"Linear : {linear_avg:.4f}")
# # #     print(f"Kalman : {kalman_avg:.4f}")

# # #     print("\nGAINS")
# # #     print("=" * 60)
# # #     print(f"Kalman - Static : {kalman_avg - static_avg:+.4f}")
# # #     print(f"Kalman - Linear : {kalman_avg - linear_avg:+.4f}")

# # #     print("\nInterpretation:")
# # #     print(
# # #         "Kalman should win when the drift is generated by a noisy latent temporal "
# # #         "process. Static fails because it does not adapt. Linear extrapolation fails "
# # #         "near curved or oscillatory drift. Kalman is more robust because it filters "
# # #         "noisy observations and uses a state-space model of the latent angle."
# # #     )


# # # # ============================================================
# # # # MAIN
# # # # ============================================================

# # # if __name__ == "__main__":
# # #     plot_dataset_examples()

# # #     results = run_experiment()

# # #     plot_results(results)

# # #     summarize_results(results)

# # #     print("\nDone.")

# # """
# # 3D Two-Moons Decision-Boundary Tracking
# # Anchor-Based Kalman Dimension-Scaling Experiment

# # Goal:
# #     Scale the same 2D anchor-output Kalman framework to 3D.

# # Pipeline:
# #     3D rotating domains
# #     -> train per-domain MLPs
# #     -> evaluate each model on fixed 3D anchors
# #     -> compress anchor-output trajectories with PCA
# #     -> predict next latent boundary state with Kalman
# #     -> reconstruct predicted anchor outputs
# #     -> adapt classifier using predicted anchor outputs
# #     -> evaluate classification accuracy

# # This does NOT estimate the rotation angle directly.
# # It tracks classifier behavior through anchor probes.
# # """

# # import os
# # import warnings
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # from copy import deepcopy
# # from sklearn.decomposition import PCA
# # from torch.utils.data import DataLoader, TensorDataset

# # from pykalman import KalmanFilter

# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt

# # warnings.filterwarnings("ignore")


# # # ============================================================
# # # SETTINGS
# # # ============================================================

# # OUT_DIR = "./kalman_3d_anchor_results"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # RANDOM_SEED = 42
# # np.random.seed(RANDOM_SEED)
# # torch.manual_seed(RANDOM_SEED)

# # T = 30
# # N_TEST = 30
# # TOTAL = T + N_TEST

# # N_SAMPLES = 300
# # NOISE = 0.10
# # DRIFT_NOISE = 0.08

# # MLP_EPOCHS = 150
# # MLP_LR = 5e-3

# # STATE_DIM = 4
# # PCA_DIM = 4
# # KF_EM_ITERS = 25

# # ADAPT_STEPS = 50
# # ADAPT_LR = 1e-3
# # ADAPT_LAMBDA = 1e-3

# # N_FOLDS = 3


# # # ============================================================
# # # DOMAIN DRIFT
# # # ============================================================

# # def chirp_theta(t):
# #     """
# #     Smooth nonlinear temporal drift.
# #     """

# #     return np.pi * np.sin(2 * np.pi * t / TOTAL)


# # def chirp_theta_noisy(t):
# #     rng = np.random.RandomState(RANDOM_SEED + t * 31)
# #     return chirp_theta(t) + rng.randn() * DRIFT_NOISE


# # # ============================================================
# # # DATASET
# # # ============================================================

# # def make_moons_3d(n_samples, noise, rng):
# #     """
# #     2D two-moons embedded in 3D with small z perturbation.
# #     """

# #     n = n_samples // 2

# #     angles0 = np.linspace(0, np.pi, n)
# #     X0 = np.stack([
# #         np.cos(angles0),
# #         np.sin(angles0),
# #         np.zeros(n)
# #     ], axis=1)

# #     angles1 = np.linspace(0, np.pi, n)
# #     X1 = np.stack([
# #         1 - np.cos(angles1),
# #         0.5 - np.sin(angles1),
# #         np.zeros(n)
# #     ], axis=1)

# #     X = np.vstack([X0, X1]).astype(np.float32)
# #     y = np.array([0] * n + [1] * n)

# #     X += rng.randn(*X.shape).astype(np.float32) * noise

# #     return X, y


# # def rotate_points_z(X, theta):
# #     R = np.array([
# #         [np.cos(theta), -np.sin(theta), 0],
# #         [np.sin(theta),  np.cos(theta), 0],
# #         [0, 0, 1]
# #     ], dtype=np.float32)

# #     return (R @ X.T).T


# # def get_dataset_3d(t, seed=RANDOM_SEED):
# #     rng = np.random.RandomState(seed + t * 100)

# #     X, y = make_moons_3d(N_SAMPLES, NOISE, rng)

# #     theta = chirp_theta_noisy(t)

# #     X = rotate_points_z(X, theta)

# #     return X, y


# # # ============================================================
# # # MODEL
# # # ============================================================

# # class MLP3D(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         self.net = nn.Sequential(
# #             nn.Linear(3, 32),
# #             nn.ReLU(),

# #             nn.Linear(32, 32),
# #             nn.ReLU(),

# #             nn.Linear(32, 2)
# #         )

# #     def forward(self, x):
# #         return self.net(x)


# # def train_mlp(X, y):
# #     model = MLP3D()

# #     loss_fn = nn.CrossEntropyLoss()
# #     opt = optim.Adam(model.parameters(), lr=MLP_LR)

# #     loader = DataLoader(
# #         TensorDataset(
# #             torch.FloatTensor(X),
# #             torch.LongTensor(y)
# #         ),
# #         batch_size=64,
# #         shuffle=True
# #     )

# #     model.train()

# #     for _ in range(MLP_EPOCHS):
# #         for xb, yb in loader:
# #             opt.zero_grad()
# #             loss = loss_fn(model(xb), yb)
# #             loss.backward()
# #             opt.step()

# #     return model


# # def evaluate(model, X, y):
# #     model.eval()

# #     with torch.no_grad():
# #         preds = model(torch.FloatTensor(X)).argmax(1).numpy()

# #     return (preds == y).mean()


# # # ============================================================
# # # 3D ANCHORS
# # # ============================================================

# # def make_anchor_grid_3d():
# #     """
# #     Boundary-sensitive 3D anchor set.

# #     This is the 3D analogue of the 2D anchor grid:
# #     anchors probe the classifier response over a fixed coordinate system.
# #     """

# #     rng = np.random.RandomState(RANDOM_SEED)

# #     anchors = []

# #     # Ring anchors in the XY plane
# #     n_ring = 72
# #     angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)

# #     for r in [0.4, 0.8, 1.2]:
# #         ring = np.stack([
# #             r * np.cos(angles),
# #             r * np.sin(angles),
# #             np.zeros(n_ring)
# #         ], axis=1)
# #         anchors.append(ring)

# #     # Slightly above/below the XY plane
# #     for z in [-0.25, 0.25]:
# #         ring = np.stack([
# #             0.8 * np.cos(angles),
# #             0.8 * np.sin(angles),
# #             np.full(n_ring, z)
# #         ], axis=1)
# #         anchors.append(ring)

# #     # Midplane random anchors near decision boundary
# #     n_mid = 96
# #     mid = rng.randn(n_mid, 3).astype(np.float32) * 0.45
# #     mid[:, 2] *= 0.25
# #     anchors.append(mid)

# #     anchors = np.vstack(anchors).astype(np.float32)

# #     print(f"Anchor count: {len(anchors)}")

# #     return anchors


# # def get_anchor_outputs(model, anchors):
# #     """
# #     Class-1 probabilities on fixed anchors.
# #     """

# #     model.eval()

# #     with torch.no_grad():
# #         logits = model(torch.FloatTensor(anchors))
# #         probs = torch.softmax(logits, dim=1)[:, 1].numpy()

# #     return probs


# # # ============================================================
# # # TEMPORAL PREDICTORS
# # # ============================================================

# # def linear_extrap_predict(obs):
# #     obs = np.asarray(obs, dtype=np.float64)
# #     obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)

# #     n_obs, obs_dim = obs.shape

# #     if n_obs < 2:
# #         return np.clip(obs[-1], 0, 1)

# #     n_comp = min(PCA_DIM, n_obs - 1, obs_dim)

# #     if n_comp < 1:
# #         return np.clip(obs[-1], 0, 1)

# #     pca = PCA(n_components=n_comp)
# #     Z = pca.fit_transform(obs)

# #     t = np.arange(n_obs)
# #     z_pred = np.zeros(n_comp)

# #     for j in range(n_comp):
# #         try:
# #             coef = np.polyfit(t, Z[:, j], 1)
# #             z_pred[j] = np.polyval(coef, n_obs)
# #         except np.linalg.LinAlgError:
# #             z_pred[j] = Z[-1, j]

# #     pred = pca.inverse_transform(z_pred.reshape(1, -1))[0]

# #     return np.clip(pred, 0, 1)

# # def pca_kalman_predict(obs):
# #     obs = np.asarray(obs, dtype=np.float64)
# #     obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)

# #     n_obs, obs_dim = obs.shape

# #     if n_obs < 6:
# #         return linear_extrap_predict(obs)

# #     n_comp = min(PCA_DIM, n_obs - 1, obs_dim)

# #     if n_comp < 2:
# #         return linear_extrap_predict(obs)

# #     pca = PCA(n_components=n_comp)
# #     Z = pca.fit_transform(obs)

# #     if n_comp < STATE_DIM:
# #         Z_state = np.hstack([
# #             Z,
# #             np.zeros((n_obs, STATE_DIM - n_comp))
# #         ])
# #     else:
# #         Z_state = Z[:, :STATE_DIM]

# #     try:
# #         kf = KalmanFilter(
# #             transition_matrices=np.eye(STATE_DIM),
# #             observation_matrices=np.eye(STATE_DIM),
# #             transition_covariance=np.eye(STATE_DIM) * 1e-3,
# #             observation_covariance=np.eye(STATE_DIM) * 1e-2,
# #             initial_state_mean=Z_state[0],
# #             initial_state_covariance=np.eye(STATE_DIM),
# #             em_vars=[
# #                 "transition_matrices",
# #                 "transition_covariance",
# #                 "observation_covariance",
# #                 "initial_state_mean",
# #                 "initial_state_covariance"
# #             ]
# #         )

# #         kf = kf.em(Z_state, n_iter=KF_EM_ITERS)

# #         means, covs = kf.filter(Z_state)

# #         next_state, _ = kf.filter_update(
# #             means[-1],
# #             covs[-1],
# #             observation=None
# #         )

# #         z_pred = next_state[:n_comp]
# #         pred = pca.inverse_transform(z_pred.reshape(1, -1))[0]

# #         return np.clip(pred, 0, 1)

# #     except Exception:
# #         return linear_extrap_predict(obs)


# # # ============================================================
# # # ADAPT MODEL TO PREDICTED ANCHOR OUTPUTS
# # # ============================================================

# # def adapt_model(base_model, anchors, target_probs):
# #     """
# #     Adapt a previous-domain classifier so that its anchor responses match
# #     predicted future anchor responses.
# #     """

# #     model = deepcopy(base_model)

# #     opt = optim.SGD(
# #         model.parameters(),
# #         lr=ADAPT_LR,
# #         momentum=0.9,
# #         nesterov=True
# #     )

# #     X_anchor = torch.FloatTensor(anchors)

# #     target = torch.FloatTensor(
# #         np.stack([
# #             1 - target_probs,
# #             target_probs
# #         ], axis=1)
# #     )

# #     old_params = [p.detach().clone() for p in model.parameters()]

# #     for _ in range(ADAPT_STEPS):
# #         model.train()
# #         opt.zero_grad()

# #         probs = torch.softmax(model(X_anchor), dim=1)

# #         loss_anchor = torch.mean((probs - target) ** 2)

# #         loss_reg = sum(
# #             torch.sum((p - p_old) ** 2)
# #             for p, p_old in zip(model.parameters(), old_params)
# #         )

# #         loss = loss_anchor + ADAPT_LAMBDA * loss_reg

# #         loss.backward()
# #         opt.step()

# #     return model


# # # ============================================================
# # # EXPERIMENT
# # # ============================================================

# # def run_single_fold(fold_seed):
# #     anchors = make_anchor_grid_3d()

# #     train_models = []
# #     observations = []

# #     print("Training per-domain MLPs...")

# #     for t in range(T):
# #         X, y = get_dataset_3d(t, seed=fold_seed)

# #         model_t = train_mlp(X, y)

# #         train_models.append(model_t)

# #         anchor_out = get_anchor_outputs(model_t, anchors)

# #         observations.append(anchor_out)

# #         print(
# #             f"  Train domain {t:2d} | "
# #             f"anchor var={np.var(anchor_out):.4f}"
# #         )

# #     print("Training static pooled model...")

# #     X_pool = []
# #     y_pool = []

# #     for t in range(T):
# #         X, y = get_dataset_3d(t, seed=fold_seed)
# #         X_pool.append(X)
# #         y_pool.append(y)

# #     X_pool = np.vstack(X_pool)
# #     y_pool = np.concatenate(y_pool)

# #     static_model = train_mlp(X_pool, y_pool)

# #     results = []

# #     for t in range(TOTAL):
# #         X, y = get_dataset_3d(t, seed=fold_seed)

# #         split = "TRAIN" if t < T else "TEST"

# #         acc_static = evaluate(static_model, X, y)

# #         if t == 0:
# #             acc_last = acc_static
# #             acc_linear = acc_static
# #             acc_kalman = acc_static

# #         elif t < T:
# #             # Use only previous domains as history
# #             obs_hist = np.array(observations[:t])

# #             base = train_models[t - 1]

# #             target_linear = linear_extrap_predict(obs_hist)
# #             target_kalman = pca_kalman_predict(obs_hist)

# #             model_linear = adapt_model(base, anchors, target_linear)
# #             model_kalman = adapt_model(base, anchors, target_kalman)

# #             acc_linear = evaluate(model_linear, X, y)
# #             acc_kalman = evaluate(model_kalman, X, y)
# #             acc_last = evaluate(base, X, y)

# #         else:
# #             # Future test domains: only training observations are available
# #             obs_hist = np.array(observations)

# #             base = train_models[-1]

# #             target_linear = linear_extrap_predict(obs_hist)
# #             target_kalman = pca_kalman_predict(obs_hist)

# #             model_linear = adapt_model(base, anchors, target_linear)
# #             model_kalman = adapt_model(base, anchors, target_kalman)

# #             acc_linear = evaluate(model_linear, X, y)
# #             acc_kalman = evaluate(model_kalman, X, y)
# #             acc_last = evaluate(base, X, y)

# #         results.append({
# #             "domain": t,
# #             "split": split,
# #             "static": acc_static,
# #             "linear": acc_linear,
# #             "kalman": acc_kalman,
# #             "last_domain": acc_last,
# #             "theta": chirp_theta_noisy(t)
# #         })

# #         print(
# #             f"Domain {t:2d} [{split}] | "
# #             f"static={acc_static:.3f} | "
# #             f"linear={acc_linear:.3f} | "
# #             f"kalman={acc_kalman:.3f} | "
# #             f"last={acc_last:.3f}"
# #         )

# #     return results


# # def run_experiment():
# #     print("=" * 70)
# #     print("3D Anchor-Based Kalman Dimension-Scaling Experiment")
# #     print("=" * 70)

# #     all_results = []

# #     fold_seeds = [
# #         RANDOM_SEED + 1000 * i
# #         for i in range(N_FOLDS)
# #     ]

# #     for i, seed in enumerate(fold_seeds):
# #         print("\n" + "=" * 50)
# #         print(f"Fold {i + 1}/{N_FOLDS} | seed={seed}")
# #         print("=" * 50)

# #         fold_results = run_single_fold(seed)
# #         all_results.append(fold_results)

# #     return all_results


# # # ============================================================
# # # SUMMARY AND PLOTS
# # # ============================================================

# # def aggregate_results(all_results):
# #     methods = ["static", "linear", "kalman", "last_domain"]

# #     avg_results = []

# #     for t in range(TOTAL):
# #         rows = [
# #             fold[t]
# #             for fold in all_results
# #         ]

# #         out = {
# #             "domain": t,
# #             "split": rows[0]["split"],
# #             "theta": np.mean([r["theta"] for r in rows])
# #         }

# #         for m in methods:
# #             vals = np.array([r[m] for r in rows])
# #             out[m] = vals.mean()
# #             out[m + "_std"] = vals.std()

# #         avg_results.append(out)

# #     return avg_results


# # def summarize(avg_results):
# #     methods = ["static", "linear", "kalman", "last_domain"]

# #     test_rows = [
# #         r for r in avg_results
# #         if r["split"] == "TEST"
# #     ]

# #     print("\n" + "=" * 70)
# #     print("TEST DOMAIN AVERAGES")
# #     print("=" * 70)

# #     for m in methods:
# #         vals = np.array([r[m] for r in test_rows])
# #         print(f"{m:12s}: {vals.mean():.4f}")


# # def plot_dataset_examples():
# #     domains = np.linspace(0, TOTAL - 1, 6, dtype=int)

# #     fig = plt.figure(figsize=(18, 10))
# #     fig.suptitle(
# #         "3D Two-Moons Domains with Sinusoidal Rotation Drift",
# #         fontsize=14,
# #         fontweight="bold"
# #     )

# #     for i, t in enumerate(domains):
# #         X, y = get_dataset_3d(t)

# #         ax = fig.add_subplot(2, 3, i + 1, projection="3d")

# #         ax.scatter(
# #             X[y == 0, 0],
# #             X[y == 0, 1],
# #             X[y == 0, 2],
# #             s=10,
# #             alpha=0.5
# #         )

# #         ax.scatter(
# #             X[y == 1, 0],
# #             X[y == 1, 1],
# #             X[y == 1, 2],
# #             s=10,
# #             alpha=0.5
# #         )

# #         split = "TRAIN" if t < T else "TEST"

# #         ax.set_title(
# #             f"Domain {t} [{split}] "
# #             f"θ={np.degrees(chirp_theta_noisy(t)):.1f}°"
# #         )

# #         ax.set_xlim(-2, 2)
# #         ax.set_ylim(-2, 2)
# #         ax.set_zlim(-1, 1)

# #         ax.set_xlabel("X")
# #         ax.set_ylabel("Y")
# #         ax.set_zlabel("Z")

# #     plt.tight_layout()

# #     out = f"{OUT_DIR}/dataset_examples.png"
# #     plt.savefig(out, dpi=150, bbox_inches="tight")
# #     plt.close()

# #     print(f"Saved dataset plot → {out}")


# # def plot_results(avg_results):
# #     domains = np.array([r["domain"] for r in avg_results])

# #     methods = ["static", "linear", "kalman", "last_domain"]

# #     labels = {
# #         "static": "Static pooled",
# #         "linear": "Linear extrapolation",
# #         "kalman": "Anchor-PCA Kalman",
# #         "last_domain": "Last domain"
# #     }

# #     plt.figure(figsize=(13, 6))

# #     plt.axvspan(0, T - 0.5, alpha=0.08, label="Train domains")
# #     plt.axvline(T - 0.5, linestyle="--", linewidth=1)

# #     for m in methods:
# #         vals = np.array([r[m] for r in avg_results])
# #         stds = np.array([r[m + "_std"] for r in avg_results])

# #         plt.plot(
# #             domains,
# #             vals,
# #             marker="o",
# #             linewidth=2,
# #             label=labels[m]
# #         )

# #         plt.fill_between(
# #             domains,
# #             vals - stds,
# #             vals + stds,
# #             alpha=0.1
# #         )

# #     plt.xlabel("Domain")
# #     plt.ylabel("Accuracy")
# #     plt.ylim(0, 1.05)

# #     plt.title(
# #         "3D Anchor-Based Decision-Boundary Tracking under Domain Drift"
# #     )

# #     plt.legend()
# #     plt.grid(alpha=0.3)

# #     plt.tight_layout()

# #     out = f"{OUT_DIR}/accuracy_results.png"
# #     plt.savefig(out, dpi=150, bbox_inches="tight")
# #     plt.close()

# #     print(f"Saved accuracy plot → {out}")


# # def save_csv(avg_results):
# #     import csv

# #     out = f"{OUT_DIR}/results.csv"

# #     keys = list(avg_results[0].keys())

# #     with open(out, "w", newline="") as f:
# #         writer = csv.DictWriter(f, fieldnames=keys)
# #         writer.writeheader()
# #         writer.writerows(avg_results)

# #     print(f"Saved CSV → {out}")


# # # ============================================================
# # # MAIN
# # # ============================================================

# # if __name__ == "__main__":
# #     plot_dataset_examples()

# #     all_results = run_experiment()

# #     avg_results = aggregate_results(all_results)

# #     summarize(avg_results)

# #     plot_results(avg_results)

# #     save_csv(avg_results)

# #     print("\nDone.")

# """
# 3D Two-Moons Anchor-Kalman Scaling Experiment
# Recursive Forecasting Version

# Goal:
#     Extend the same anchor-based Kalman decision-boundary tracking framework
#     from 2D to 3D.

# Important fix:
#     During test time, Kalman predictions are rolled forward recursively.
#     The model no longer uses the same one-step prediction for every test domain.
# """

# import os
# import csv
# import warnings
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from copy import deepcopy
# from sklearn.decomposition import PCA
# from torch.utils.data import DataLoader, TensorDataset
# from pykalman import KalmanFilter

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore")


# # ============================================================
# # SETTINGS
# # ============================================================

# OUT_DIR = "./kalman_3d_recursive_results"
# os.makedirs(OUT_DIR, exist_ok=True)

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)

# T = 40
# N_TEST = 20
# TOTAL = T + N_TEST

# N_SAMPLES = 600
# NOISE = 0.10
# DRIFT_NOISE = 0.06

# MLP_EPOCHS = 250
# MLP_LR = 5e-3

# STATE_DIM = 6
# PCA_DIM = 6
# KF_EM_ITERS = 30

# ADAPT_STEPS = 250
# ADAPT_LR = 3e-3
# ADAPT_LAMBDA = 1e-4

# N_FOLDS = 3


# # ============================================================
# # DOMAIN DRIFT
# # ============================================================

# def chirp_theta(t):
#     return np.pi * np.sin(2 * np.pi * t / TOTAL)


# def chirp_theta_noisy(t):
#     rng = np.random.RandomState(RANDOM_SEED + t * 31)
#     return chirp_theta(t) + rng.randn() * DRIFT_NOISE


# # ============================================================
# # DATASET
# # ============================================================

# def make_moons_3d(n_samples, noise, rng):
#     n = n_samples // 2

#     angles0 = np.linspace(0, np.pi, n)
#     X0 = np.stack([
#         np.cos(angles0),
#         np.sin(angles0),
#         np.zeros(n)
#     ], axis=1)

#     angles1 = np.linspace(0, np.pi, n)
#     X1 = np.stack([
#         1 - np.cos(angles1),
#         0.5 - np.sin(angles1),
#         np.zeros(n)
#     ], axis=1)

#     X = np.vstack([X0, X1]).astype(np.float32)
#     y = np.array([0] * n + [1] * n)

#     X += rng.randn(*X.shape).astype(np.float32) * noise

#     return X, y


# def rotate_points_z(X, theta):
#     R = np.array([
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta),  np.cos(theta), 0],
#         [0, 0, 1]
#     ], dtype=np.float32)

#     return (R @ X.T).T


# def get_dataset_3d(t, seed=RANDOM_SEED):
#     rng = np.random.RandomState(seed + t * 100)

#     X, y = make_moons_3d(N_SAMPLES, NOISE, rng)
#     theta = chirp_theta_noisy(t)

#     X = rotate_points_z(X, theta)

#     return X, y


# # ============================================================
# # MODEL
# # ============================================================

# class MLP3D(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(3, 64),
#             nn.ReLU(),

#             nn.Linear(64, 64),
#             nn.ReLU(),

#             nn.Linear(64, 32),
#             nn.ReLU(),

#             nn.Linear(32, 2)
#         )

#     def forward(self, x):
#         return self.net(x)


# def train_mlp(X, y):
#     model = MLP3D()

#     loss_fn = nn.CrossEntropyLoss()
#     opt = optim.Adam(model.parameters(), lr=MLP_LR)

#     loader = DataLoader(
#         TensorDataset(
#             torch.FloatTensor(X),
#             torch.LongTensor(y)
#         ),
#         batch_size=128,
#         shuffle=True
#     )

#     model.train()

#     for _ in range(MLP_EPOCHS):
#         for xb, yb in loader:
#             opt.zero_grad()
#             loss = loss_fn(model(xb), yb)
#             loss.backward()
#             opt.step()

#     return model


# def evaluate(model, X, y):
#     model.eval()

#     with torch.no_grad():
#         preds = model(torch.FloatTensor(X)).argmax(1).numpy()

#     return (preds == y).mean()


# # ============================================================
# # 3D ANCHORS
# # ============================================================

# def make_anchor_grid_3d():
#     rng = np.random.RandomState(RANDOM_SEED)

#     anchors = []

#     n_ring = 96
#     angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)

#     # Dense XY-plane rings
#     for r in [0.25, 0.45, 0.70, 0.95, 1.20, 1.45]:
#         ring = np.stack([
#             r * np.cos(angles),
#             r * np.sin(angles),
#             np.zeros(n_ring)
#         ], axis=1)
#         anchors.append(ring)

#     # Slightly above and below the XY plane
#     for z in [-0.30, -0.15, 0.15, 0.30]:
#         for r in [0.55, 0.95, 1.35]:
#             ring = np.stack([
#                 r * np.cos(angles),
#                 r * np.sin(angles),
#                 np.full(n_ring, z)
#             ], axis=1)
#             anchors.append(ring)

#     # Random boundary-region anchors
#     n_mid = 300
#     mid = rng.randn(n_mid, 3).astype(np.float32) * 0.55
#     mid[:, 2] *= 0.30
#     anchors.append(mid)

#     anchors = np.vstack(anchors).astype(np.float32)

#     print(f"Anchor count: {len(anchors)}")

#     return anchors


# def get_anchor_outputs(model, anchors):
#     model.eval()

#     with torch.no_grad():
#         logits = model(torch.FloatTensor(anchors))
#         probs = torch.softmax(logits, dim=1)[:, 1].numpy()

#     probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

#     return np.clip(probs, 0, 1)


# # ============================================================
# # KALMAN IN PCA ANCHOR SPACE
# # ============================================================

# def pca_kalman_predict(obs):
#     obs = np.asarray(obs, dtype=np.float64)
#     obs = np.nan_to_num(obs, nan=0.5, posinf=1.0, neginf=0.0)

#     n_obs, obs_dim = obs.shape

#     if n_obs < 6:
#         return np.clip(obs[-1], 0, 1)

#     n_comp = min(PCA_DIM, n_obs - 1, obs_dim)

#     if n_comp < 2:
#         return np.clip(obs[-1], 0, 1)

#     pca = PCA(n_components=n_comp)
#     Z = pca.fit_transform(obs)

#     if n_comp < STATE_DIM:
#         Z_state = np.hstack([
#             Z,
#             np.zeros((n_obs, STATE_DIM - n_comp))
#         ])
#     else:
#         Z_state = Z[:, :STATE_DIM]

#     try:
#         kf = KalmanFilter(
#             transition_matrices=np.eye(STATE_DIM),
#             observation_matrices=np.eye(STATE_DIM),
#             transition_covariance=np.eye(STATE_DIM) * 1e-3,
#             observation_covariance=np.eye(STATE_DIM) * 5e-3,
#             initial_state_mean=Z_state[0],
#             initial_state_covariance=np.eye(STATE_DIM),
#             em_vars=[
#                 "transition_matrices",
#                 "transition_covariance",
#                 "observation_covariance",
#                 "initial_state_mean",
#                 "initial_state_covariance"
#             ]
#         )

#         kf = kf.em(Z_state, n_iter=KF_EM_ITERS)
#         means, covs = kf.filter(Z_state)

#         next_state, _ = kf.filter_update(
#             means[-1],
#             covs[-1],
#             observation=None
#         )

#         z_pred = next_state[:n_comp]
#         pred = pca.inverse_transform(z_pred.reshape(1, -1))[0]

#         return np.clip(pred, 0, 1)

#     except Exception:
#         return np.clip(obs[-1], 0, 1)


# # ============================================================
# # ADAPTATION
# # ============================================================

# def adapt_model(base_model, anchors, target_probs):
#     model = deepcopy(base_model)

#     opt = optim.SGD(
#         model.parameters(),
#         lr=ADAPT_LR,
#         momentum=0.9,
#         nesterov=True
#     )

#     X_anchor = torch.FloatTensor(anchors)

#     target = torch.FloatTensor(
#         np.stack([
#             1 - target_probs,
#             target_probs
#         ], axis=1)
#     )

#     old_params = [p.detach().clone() for p in model.parameters()]

#     for _ in range(ADAPT_STEPS):
#         model.train()
#         opt.zero_grad()

#         probs = torch.softmax(model(X_anchor), dim=1)

#         loss_anchor = torch.mean((probs - target) ** 2)

#         loss_reg = sum(
#             torch.sum((p - p_old) ** 2)
#             for p, p_old in zip(model.parameters(), old_params)
#         )

#         loss = loss_anchor + ADAPT_LAMBDA * loss_reg

#         loss.backward()
#         opt.step()

#     return model


# # ============================================================
# # EXPERIMENT
# # ============================================================

# def run_single_fold(fold_seed):
#     anchors = make_anchor_grid_3d()

#     train_models = []
#     observations = []

#     print("Training per-domain MLPs...")

#     for t in range(T):
#         X, y = get_dataset_3d(t, seed=fold_seed)

#         model_t = train_mlp(X, y)
#         train_models.append(model_t)

#         anchor_out = get_anchor_outputs(model_t, anchors)
#         observations.append(anchor_out)

#         print(
#             f"  Train domain {t:2d} | "
#             f"anchor var={np.var(anchor_out):.4f}"
#         )

#     results = []

#     # Recursive test-time forecast history
#     forecast_history = [obs.copy() for obs in observations]
#     base_test_model = train_models[-1]

#     for t in range(TOTAL):
#         X, y = get_dataset_3d(t, seed=fold_seed)

#         split = "TRAIN" if t < T else "TEST"

#         if t == 0:
#             model_kalman = train_models[0]
#             acc_kalman = evaluate(model_kalman, X, y)

#         elif t < T:
#             # Use previous observed classifiers only
#             obs_hist = np.array(observations[:t])
#             base = train_models[t - 1]

#             target_kalman = pca_kalman_predict(obs_hist)
#             model_kalman = adapt_model(base, anchors, target_kalman)

#             acc_kalman = evaluate(model_kalman, X, y)

#         else:
#             # Online supervised setting:
#             # 1. Predict/adapt/evaluate current test domain.
#             # 2. Then train on current test domain and add its real anchor output
#             #    to the history for future predictions.

#             obs_hist = np.array(forecast_history)

#             target_kalman = pca_kalman_predict(obs_hist)

#             model_kalman = adapt_model(
#                 base_test_model,
#                 anchors,
#                 target_kalman
#             )

#             acc_kalman = evaluate(model_kalman, X, y)

#             # After evaluation, current test domain becomes observed.
#             # This is what should happen in an online/continual setting.
#             model_t = train_mlp(X, y)
#             real_anchor_out = get_anchor_outputs(model_t, anchors)

#             forecast_history.append(real_anchor_out.copy())
#             base_test_model = model_t

#         results.append({
#             "domain": t,
#             "split": split,
#             "kalman": acc_kalman,
#             "theta": chirp_theta_noisy(t)
#         })

#         print(
#             f"Domain {t:2d} [{split}] | "
#             f"kalman={acc_kalman:.3f}"
#         )

#     return results


# def run_experiment():
#     print("=" * 70)
#     print("3D Anchor-Based Kalman Scaling Experiment")
#     print("Recursive Forecasting Version")
#     print("=" * 70)

#     all_results = []

#     fold_seeds = [
#         RANDOM_SEED + 1000 * i
#         for i in range(N_FOLDS)
#     ]

#     for i, seed in enumerate(fold_seeds):
#         print("\n" + "=" * 50)
#         print(f"Fold {i + 1}/{N_FOLDS} | seed={seed}")
#         print("=" * 50)

#         fold_results = run_single_fold(seed)
#         all_results.append(fold_results)

#     return all_results


# # ============================================================
# # SUMMARY AND PLOTS
# # ============================================================

# def aggregate_results(all_results):
#     avg_results = []

#     for t in range(TOTAL):
#         rows = [fold[t] for fold in all_results]
#         vals = np.array([r["kalman"] for r in rows])

#         avg_results.append({
#             "domain": t,
#             "split": rows[0]["split"],
#             "theta": np.mean([r["theta"] for r in rows]),
#             "kalman": vals.mean(),
#             "kalman_std": vals.std()
#         })

#     return avg_results


# def summarize(avg_results):
#     test_rows = [r for r in avg_results if r["split"] == "TEST"]
#     test_vals = np.array([r["kalman"] for r in test_rows])

#     print("\n" + "=" * 70)
#     print("KALMAN TEST PERFORMANCE")
#     print("=" * 70)
#     print(f"Mean test accuracy : {test_vals.mean():.4f}")
#     print(f"Std across domains : {test_vals.std():.4f}")
#     print(f"Min test accuracy  : {test_vals.min():.4f}")
#     print(f"Max test accuracy  : {test_vals.max():.4f}")


# def plot_dataset_examples():
#     domains = np.linspace(0, TOTAL - 1, 6, dtype=int)

#     fig = plt.figure(figsize=(18, 10))
#     fig.suptitle(
#         "3D Two-Moons Domains with Sinusoidal Rotation Drift",
#         fontsize=14,
#         fontweight="bold"
#     )

#     for i, t in enumerate(domains):
#         X, y = get_dataset_3d(t)

#         ax = fig.add_subplot(2, 3, i + 1, projection="3d")

#         ax.scatter(
#             X[y == 0, 0],
#             X[y == 0, 1],
#             X[y == 0, 2],
#             s=10,
#             alpha=0.5
#         )

#         ax.scatter(
#             X[y == 1, 0],
#             X[y == 1, 1],
#             X[y == 1, 2],
#             s=10,
#             alpha=0.5
#         )

#         split = "TRAIN" if t < T else "TEST"

#         ax.set_title(
#             f"Domain {t} [{split}] "
#             f"theta={np.degrees(chirp_theta_noisy(t)):.1f} deg"
#         )

#         ax.set_xlim(-2, 2)
#         ax.set_ylim(-2, 2)
#         ax.set_zlim(-1, 1)

#     plt.tight_layout()

#     out = f"{OUT_DIR}/dataset_examples.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"Saved dataset plot → {out}")


# def plot_results(avg_results):
#     domains = np.array([r["domain"] for r in avg_results])
#     vals = np.array([r["kalman"] for r in avg_results])
#     stds = np.array([r["kalman_std"] for r in avg_results])

#     plt.figure(figsize=(13, 6))

#     plt.axvspan(0, T - 0.5, alpha=0.08, label="Observed training domains")
#     plt.axvline(T - 0.5, linestyle="--", linewidth=1)

#     plt.plot(
#         domains,
#         vals,
#         marker="o",
#         linewidth=2,
#         label="Recursive Kalman-guided adaptation"
#     )

#     plt.fill_between(
#         domains,
#         vals - stds,
#         vals + stds,
#         alpha=0.15
#     )

#     plt.xlabel("Domain")
#     plt.ylabel("Classification Accuracy")
#     plt.ylim(0, 1.05)

#     plt.title("3D Anchor-Based Recursive Kalman Adaptation Accuracy")

#     plt.legend()
#     plt.grid(alpha=0.3)

#     plt.tight_layout()

#     out = f"{OUT_DIR}/kalman_accuracy_results.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()

#     print(f"Saved accuracy plot → {out}")


# def save_csv(avg_results):
#     out = f"{OUT_DIR}/kalman_results.csv"

#     keys = list(avg_results[0].keys())

#     with open(out, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         writer.writerows(avg_results)

#     print(f"Saved CSV → {out}")


# # ============================================================
# # MAIN
# # ============================================================

# if __name__ == "__main__":
#     plot_dataset_examples()

#     all_results = run_experiment()

#     avg_results = aggregate_results(all_results)

#     summarize(avg_results)

#     plot_results(avg_results)

#     save_csv(avg_results)

#     print("\nDone.")