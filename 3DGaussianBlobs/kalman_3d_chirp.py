"""
3D Gaussian Blobs with Chirp Drift — Kalman Anchor Point Method (Fixed)
Key fix: rolling Kalman — at test domain t, predict using obs 0..t-1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pykalman import KalmanFilter
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUT_DIR     = '/scratch4/en580/vlu_code/final_grind'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

T           = 50
N_TEST      = 6
N_SAMPLES   = 400
NOISE       = 0.42
GRID_SIZE   = 6       # 6^3=216 anchors instead of 8^3=512 — faster
STATE_DIM   = 10
KF_EM_ITERS = 10
MLP_EPOCHS  = 300     # reduced from 800
MLP_LR      = 5e-3
F0          = 0.0050
F1          = 0.0440
TOTAL       = T + N_TEST
N_FOLDS     = 3       # reduced from 5

DRIFT_NOISE = 0.45
SEP         = 0.85


def chirp_theta(t):
    """Base chirp — smooth underlying trend."""
    return 2*np.pi*(F0*t + 0.5*(F1-F0)*t**2/(TOTAL-1))


def chirp_theta_noisy(t):
    """Chirp + domain-specific noise — what each domain actually observes."""
    base  = chirp_theta(t)
    rng   = np.random.RandomState(RANDOM_SEED + t * 31)
    noise = rng.randn() * DRIFT_NOISE
    return base + noise


def get_dataset_3d(t, seed=RANDOM_SEED):
    rng   = np.random.RandomState(seed + t*100)
    theta = chirp_theta_noisy(t)   # noisy drift per domain
    tilt  = theta * 0.3
    c0    = np.array([SEP, 0.0, 0.0])
    c1    = np.array([-SEP, 0.0, 0.0])
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta), np.cos(theta),0],[0,0,1]])
    Rx = np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],
                   [0,np.sin(tilt),np.cos(tilt)]])
    R   = Rx @ Rz
    c0r = R @ c0; c1r = R @ c1
    n   = N_SAMPLES // 2
    X   = np.vstack([rng.randn(n,3)*NOISE+c0r,
                     rng.randn(n,3)*NOISE+c1r]).astype(np.float32)
    y   = np.array([0]*n + [1]*n)
    return X, y


class MLP3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3,32), nn.ReLU(),
            nn.Linear(32,32), nn.ReLU(),
            nn.Linear(32,2)
        )
    def forward(self, x): return self.net(x)


def train_mlp(X, y, epochs=MLP_EPOCHS):
    model   = MLP3D()
    loss_fn = nn.CrossEntropyLoss()
    opt     = optim.Adam(model.parameters(), lr=MLP_LR)
    loader  = DataLoader(TensorDataset(torch.FloatTensor(X),
                                       torch.LongTensor(y)),
                         batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X)).argmax(1).numpy()
    return (preds == y).mean()


def get_anchor_outputs(model, anchors):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.FloatTensor(anchors)), dim=-1)[:,1].numpy()
    return probs


def make_anchor_grid():
    lin = np.linspace(-2.0, 2.0, GRID_SIZE)
    g1,g2,g3 = np.meshgrid(lin, lin, lin)
    return np.stack([g1.ravel(),g2.ravel(),g3.ravel()],axis=1).astype(np.float32)


def kalman_predict(observations, forgetting=None):
    """Standard Kalman — all observations weighted equally."""
    obs_dim = observations.shape[1]
    np.random.seed(RANDOM_SEED)
    kf = KalmanFilter(
        np.eye(STATE_DIM),
        np.random.uniform(-1,1,(obs_dim,STATE_DIM)),
        np.diag(np.random.uniform(0,1,STATE_DIM)),
        np.diag(np.random.uniform(0,1,obs_dim)),
        np.zeros(STATE_DIM), np.zeros(obs_dim),
        np.zeros(STATE_DIM), np.eye(STATE_DIM),
        em_vars=['transition_matrices','observation_matrices',
                 'transition_covariance','observation_covariance',
                 'initial_state_mean','initial_state_covariance']
    )
    kf.em(X=observations, n_iter=KF_EM_ITERS)
    means, covs = kf.filter(observations)
    ns, _ = kf.filter_update(means[-1], covs[-1], observation=None)
    return np.clip(kf.observation_matrices @ ns, 0, 1)


def kalman_predict_weighted(observations, forgetting=0.80):
    """
    Forgetting-factor Kalman — older observations treated as noisier.
    For observation at time t in a sequence of T_obs:
      R_t = R_base / lambda^(T_obs - 1 - t)
    So the most recent observation has R_base, older ones have larger R
    (= less trusted). lambda < 1 means older = noisier = ignored more.
    """
    obs_dim = observations.shape[1]
    T_obs   = len(observations)
    np.random.seed(RANDOM_SEED)

    # Build time-varying observation noise covariances
    # R_t = base_var * (1/lambda)^(T_obs-1-t)  → older = higher variance
    base_var   = np.random.uniform(0, 1, obs_dim)
    obs_covs   = []
    for t in range(T_obs):
        scale    = (1.0 / forgetting) ** (T_obs - 1 - t)
        obs_covs.append(np.diag(base_var * scale))

    # Use standard KF for EM to learn transition/observation matrices
    kf = KalmanFilter(
        np.eye(STATE_DIM),
        np.random.uniform(-1,1,(obs_dim,STATE_DIM)),
        np.diag(np.random.uniform(0,1,STATE_DIM)),
        np.diag(base_var),          # base R for EM
        np.zeros(STATE_DIM), np.zeros(obs_dim),
        np.zeros(STATE_DIM), np.eye(STATE_DIM),
        em_vars=['transition_matrices','observation_matrices',
                 'transition_covariance','observation_covariance',
                 'initial_state_mean','initial_state_covariance']
    )
    kf.em(X=observations, n_iter=KF_EM_ITERS)

    # Manual forward filter with time-varying R
    x = kf.initial_state_mean.copy()
    P = kf.initial_state_covariance.copy()
    A = kf.transition_matrices
    C = kf.observation_matrices
    Q = kf.transition_covariance

    for t, obs in enumerate(observations):
        # Predict
        x = A @ x
        P = A @ P @ A.T + Q
        # Update with time-varying R
        R_t = obs_covs[t]
        S   = C @ P @ C.T + R_t
        K   = P @ C.T @ np.linalg.solve(S, np.eye(obs_dim)).T
        x   = x + K @ (obs - C @ x)
        P   = (np.eye(STATE_DIM) - K @ C) @ P

    # Predict next step
    x_next = A @ x
    return np.clip(C @ x_next, 0, 1)


def linear_extrap_predict(observations):
    """
    Linear extrapolation of anchor outputs.
    For each anchor point, fit a linear trend over time and extrapolate
    one step forward. No latent state, no Kalman — just numpy polyfit.
    """
    T_obs   = len(observations)
    obs_dim = observations.shape[1]
    t_vals  = np.arange(T_obs, dtype=np.float64)
    preds   = np.zeros(obs_dim)
    for d in range(obs_dim):
        # Fit degree-1 polynomial (line) through observations[:,d]
        coeffs   = np.polyfit(t_vals, observations[:, d], deg=1)
        preds[d] = np.polyval(coeffs, T_obs)   # extrapolate to t=T_obs
    return np.clip(preds, 0, 1)


def lstm_predict_weights(model_list):
    """
    LSTM trained on weight sequences to predict next domain weights.
    No anchors — operates directly in weight space.
    Input: sequence of flattened weight vectors from model_list
    Output: new model with predicted weights loaded
    """
    def flatten_weights(m):
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    weight_seqs = torch.stack([flatten_weights(m) for m in model_list])
    param_dim   = weight_seqs.shape[1]
    hidden      = 64

    lstm = nn.LSTM(param_dim, hidden, num_layers=1, batch_first=True)
    head = nn.Linear(hidden, param_dim)
    opt  = optim.Adam(list(lstm.parameters()) +
                      list(head.parameters()), lr=1e-3)

    # Train: given weights at 0..t-1, predict weights at 1..t
    X_seq = weight_seqs[:-1].unsqueeze(0)   # (1, T-1, param_dim)
    y_seq = weight_seqs[1:].unsqueeze(0)    # (1, T-1, param_dim)

    for _ in range(200):
        opt.zero_grad()
        out, _ = lstm(X_seq)
        loss   = nn.MSELoss()(head(out), y_seq)
        loss.backward()
        opt.step()

    # Predict next step using full sequence
    with torch.no_grad():
        out, _ = lstm(weight_seqs.unsqueeze(0))
        next_w = head(out[0, -1])

    # Load predicted weights into a new model
    model   = deepcopy(model_list[-1])
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(next_w[pointer:pointer+n].reshape(p.shape))
        pointer += n
    return model


def adapt_model(base, anchors, target_probs, lam=1e-3, steps=150):
    model = deepcopy(base)
    opt   = optim.SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.9)
    Xp    = torch.FloatTensor(anchors)
    tgt   = torch.FloatTensor(
        np.stack([1-target_probs, target_probs], axis=1)).flatten()
    old_p = [p.detach().clone() for p in model.parameters()]
    for _ in range(steps):
        model.train(); opt.zero_grad()
        out  = torch.softmax(model(Xp), dim=-1).flatten()
        loss = torch.mean((out-tgt)**2) + \
               lam*sum(torch.sum((p-po)**2)
                       for p,po in zip(model.parameters(), old_p))
        loss.backward(); opt.step()
    return model


def run_single_fold(fold_seed):
    """Run one fold with a specific random seed."""
    all_domains   = list(range(TOTAL))
    train_domains = list(range(T))
    anchors       = make_anchor_grid()

    # Per-domain MLPs
    model_list   = []
    observations = []
    for t in train_domains:
        X, y = get_dataset_3d(t, seed=fold_seed)
        m    = train_mlp(X, y)
        model_list.append(m)
        observations.append(get_anchor_outputs(m, anchors))

    # Static pooled
    X_pool = np.vstack([get_dataset_3d(t, seed=fold_seed)[0] for t in train_domains])
    y_pool = np.concatenate([get_dataset_3d(t, seed=fold_seed)[1] for t in train_domains])
    static_model = train_mlp(X_pool, y_pool, epochs=MLP_EPOCHS)

    results = []
    for t in all_domains:
        X, y  = get_dataset_3d(t, seed=fold_seed)
        split = 'train' if t < T else 'test'
        acc_s = evaluate(static_model, X, y)

        if t < 2:
            acc_k    = acc_s
            acc_kw   = acc_s
            acc_le   = acc_s
            acc_lstm = acc_s
        else:
            n_obs = min(t, len(observations))
            obs   = np.array(observations[:n_obs])
            base  = model_list[min(t-1, len(model_list)-1)]

            # Standard Kalman
            preds  = kalman_predict(obs)
            acc_k  = evaluate(adapt_model(base, anchors, preds), X, y)

            # Weighted Kalman
            preds_w = kalman_predict_weighted(obs, forgetting=0.80)
            acc_kw  = evaluate(adapt_model(base, anchors, preds_w), X, y)

            # Linear extrapolation
            preds_le = linear_extrap_predict(obs)
            acc_le   = evaluate(adapt_model(base, anchors, preds_le), X, y)

            # LSTM on weight space
            lstm_m   = lstm_predict_weights(model_list[:t])
            acc_lstm = evaluate(lstm_m, X, y)

        if t == 0:
            acc_l = acc_s
        else:
            Xl, yl = get_dataset_3d(t-1, seed=fold_seed)
            acc_l  = evaluate(train_mlp(Xl, yl), X, y)

        results.append({'domain':t,'theta':chirp_theta_noisy(t),
                        'split':split,
                        'static':acc_s,'kalman':acc_k,'kalman_w':acc_kw,
                        'linear_extrap':acc_le,'lstm':acc_lstm,
                        'last_domain':acc_l})
    return results


def run_experiment():
    print("="*60)
    print(f"3D Chirp Drift — {N_FOLDS}-Fold CV")
    print("="*60)

    print("Chirp θ schedule:")
    for t in range(TOTAL):
        tag = 'TRAIN' if t<T else 'TEST '
        print(f"  Domain {t:2d} [{tag}]: "
              f"θ_smooth={np.degrees(chirp_theta(t)):.1f}°  "
              f"θ_actual={np.degrees(chirp_theta_noisy(t)):.1f}°")

    # Run N_FOLDS with different seeds
    fold_seeds = [RANDOM_SEED + f*1000 for f in range(N_FOLDS)]
    all_fold_results = []

    for fi, seed in enumerate(fold_seeds):
        print(f"\n--- Fold {fi+1}/{N_FOLDS} (seed={seed}) ---")
        fold_results = run_single_fold(seed)
        all_fold_results.append(fold_results)

        test_r = [r for r in fold_results if r['split']=='test']
        for m in ['static','linear_extrap','lstm','kalman','kalman_w','last_domain']:
            avg = np.mean([r[m] for r in test_r])
            print(f"  {m:14s}: {avg:.4f}")

    # Average across folds per domain
    all_domains = list(range(TOTAL))
    avg_results = []
    for t in all_domains:
        domain_rows = [[r for r in fold if r['domain']==t][0]
                       for fold in all_fold_results]
        avg_results.append({
            'domain': t,
            'theta':  np.mean([r['theta'] for r in domain_rows]),
            'split':  domain_rows[0]['split'],
            **{m: np.mean([r[m] for r in domain_rows])
               for m in ['static','kalman','kalman_w',
                          'linear_extrap','lstm','last_domain']},
            **{f'{m}_std': np.std([r[m] for r in domain_rows])
               for m in ['static','kalman','kalman_w',
                          'linear_extrap','lstm','last_domain']},
        })

    df = {k:[r[k] for r in avg_results] for k in avg_results[0]}

    print("\n--- CV Test domain averages (mean ± std) ---")
    test_r = [r for r in avg_results if r['split']=='test']
    for m in ['static','linear_extrap','lstm','kalman','kalman_w','last_domain']:
        vals = [r[m] for r in test_r]
        stds = [r[f'{m}_std'] for r in test_r]
        print(f"  {m:14s}: {np.mean(vals):.4f} ± {np.mean(stds):.4f}")

    # For visualization use first fold's model_list
    anchors = make_anchor_grid()
    model_list = []
    for t in range(T):
        X, y = get_dataset_3d(t, seed=RANDOM_SEED)
        model_list.append(train_mlp(X, y))

    X_pool = np.vstack([get_dataset_3d(t, seed=RANDOM_SEED)[0] for t in range(T)])
    y_pool = np.concatenate([get_dataset_3d(t, seed=RANDOM_SEED)[1] for t in range(T)])
    static_model = train_mlp(X_pool, y_pool, epochs=MLP_EPOCHS)

    return df, model_list, static_model


def plot_results(df, model_list, static_model):
    fig = plt.figure(figsize=(18,11))
    fig.suptitle(
        f'Kalman Anchor Point Method — 3D Gaussian Blobs with Chirp Drift\n'
        f'Slow drift early → Accelerating drift → Unseen test domains  ({N_FOLDS}-fold CV, mean ± std)',
        fontsize=13, fontweight='bold'
    )
    domains   = df['domain']
    thetas    = df['theta']
    TRAIN_END = T - 0.5

    show_t    = [0,4,9]
    subtitles = ['Early (slow drift)','Mid (medium drift)','Late (fast drift)']
    for i,(t,sub) in enumerate(zip(show_t,subtitles)):
        ax = fig.add_subplot(2,4,i+1,projection='3d')
        X,y = get_dataset_3d(t)
        ax.scatter(X[y==0,0],X[y==0,1],X[y==0,2],c='#378ADD',s=8,alpha=0.7)
        ax.scatter(X[y==1,0],X[y==1,1],X[y==1,2],c='#E24B4A',s=8,alpha=0.7)
        ax.set_title(f'Domain {t}\n{sub}',fontsize=8)
        ax.tick_params(labelsize=5)
        ax.set_xlabel('X',fontsize=6); ax.set_ylabel('Y',fontsize=6)
        ax.set_zlabel('Z',fontsize=6)

    ax = fig.add_subplot(2,4,4,projection='3d')
    X,y = get_dataset_3d(T)
    ax.scatter(X[y==0,0],X[y==0,1],X[y==0,2],c='#378ADD',s=8,alpha=0.7)
    ax.scatter(X[y==1,0],X[y==1,1],X[y==1,2],c='#E24B4A',s=8,alpha=0.7)
    ax.set_title(f'Domain {T} (TEST)\nUnseen domain',fontsize=8,color='#E24B4A')
    ax.tick_params(labelsize=5)
    ax.set_xlabel('X',fontsize=6); ax.set_ylabel('Y',fontsize=6)
    ax.set_zlabel('Z',fontsize=6)

    ax5 = fig.add_subplot(2,4,5)
    ax5.axvspan(0,TRAIN_END,alpha=0.08,color='steelblue')
    ax5.axvspan(TRAIN_END,max(domains)+0.5,alpha=0.08,color='tomato')
    ax5.axvline(TRAIN_END,color='gray',linestyle=':',linewidth=1)
    # Smooth underlying trend
    smooth_thetas = [chirp_theta(t) for t in domains]
    ax5.plot(domains,smooth_thetas,'--',color='#378ADD',linewidth=1.5,
             alpha=0.5,label='Smooth trend')
    # Actual noisy thetas
    ax5.plot(domains,thetas,'o-',color='#378ADD',linewidth=2,markersize=6,
             label='Actual θ (noisy)')
    ax5.fill_between(domains,thetas,alpha=0.12,color='#378ADD')
    ax5.set_xlabel('Domain',fontsize=9); ax5.set_ylabel('θ (rad)',fontsize=9)
    ax5.set_title('Chirp drift θ(t)\n(noisy — last-domain unreliable)',fontsize=9)
    ax5.grid(True,alpha=0.25)
    ax5.legend(fontsize=7,loc='upper left')
    ax5.text(2,max(thetas)*0.12,'Training',fontsize=8,color='steelblue')
    ax5.text(T+0.3,max(thetas)*0.12,'Test',fontsize=8,color='tomato')

    ax6 = fig.add_subplot(2,4,6)
    ax6.axvspan(0,TRAIN_END,alpha=0.08,color='steelblue')
    ax6.axvspan(TRAIN_END,max(domains)+0.5,alpha=0.08,color='tomato')
    ax6.axvline(TRAIN_END,color='gray',linestyle=':',linewidth=1)
    methods = [
        ('static',       '#2171b5', 'o-',  'Static (pooled)'),
        ('linear_extrap','#f16913', 'v-',  'Linear extrapolation (anchor)'),
        ('lstm',         '#756bb1', 'p-',  'LSTM (weight space)'),
        ('kalman',       '#cb181d', '^-',  'Kalman uniform (ours)'),
        ('kalman_w',     '#8B008B', 's-',  'Kalman weighted λ=0.80 (ours)'),
        ('last_domain',  '#238b45', 'D--', 'Trained on domain t-1'),
    ]
    for key, color, fmt, label in methods:
        ax6.plot(domains, df[key], fmt, color=color, linewidth=2,
                 markersize=5, label=label)
        ax6.fill_between(domains,
                         [s-e for s,e in zip(df[key], df[f'{key}_std'])],
                         [s+e for s,e in zip(df[key], df[f'{key}_std'])],
                         alpha=0.08, color=color)

    # Shade best Kalman gain over static in test period
    best_kalman = [max(k,kw) for k,kw in zip(df['kalman'],df['kalman_w'])]
    td = [d for d,s in zip(domains,df['split']) if s=='test']
    ts = [v for v,s in zip(df['static'],df['split']) if s=='test']
    tk = [max(k,kw) for k,kw,s in
          zip(df['kalman'],df['kalman_w'],df['split']) if s=='test']
    ax6.fill_between(td,ts,tk,where=[k>=s for k,s in zip(tk,ts)],
                     alpha=0.12,color='purple',label='Kalman gain (best)')
    ax6.set_xlabel('Domain',fontsize=9); ax6.set_ylabel('Accuracy',fontsize=9)
    ax6.set_title('Per-domain accuracy',fontsize=9)
    ax6.set_ylim(0.0,1.05); ax6.legend(fontsize=7,loc='lower left')
    ax6.grid(True,alpha=0.25)

    ax7 = fig.add_subplot(2,4,7)
    tr   = [r for r in [dict(zip(df.keys(),[df[k][i] for k in df]))
                         for i in range(len(domains))] if r['split']=='test']
    all_m   = ['static','linear_extrap','lstm','kalman','kalman_w','last_domain']
    labels_b = ['Static','Linear\nextrap','LSTM','Kalman\nuniform',
                'Kalman\nweighted','Last\ndomain']
    colors_b = ['#4292c6','#f16913','#756bb1','#cb181d','#8B008B','#41ab5d']
    avgs = {m:np.mean([r[m] for r in tr]) for m in all_m}
    bars = ax7.bar(labels_b,[avgs[m] for m in all_m],
                   color=colors_b,width=0.6,edgecolor='white',linewidth=1.5)
    for bar,m in zip(bars,all_m):
        v = avgs[m]
        ax7.text(bar.get_x()+bar.get_width()/2, v+0.006,
                 f'{v:.3f}',ha='center',va='bottom',fontsize=8,fontweight='bold')
    best_k   = max(['kalman','kalman_w'],key=lambda m:avgs[m])
    best_idx = all_m.index(best_k)
    bars[best_idx].set_edgecolor('#4B0082'); bars[best_idx].set_linewidth(2.5)
    gain = avgs[best_k]-avgs['static']
    sign = '+' if gain>=0 else ''
    ax7.annotate(f'{sign}{gain:.3f}\nvs static',
                 xy=(best_idx,avgs[best_k]),
                 xytext=(best_idx,min(avgs[best_k]+0.09,1.08)),
                 ha='center',fontsize=8,color='#4B0082',fontweight='bold',
                 arrowprops=dict(arrowstyle='->',color='#4B0082'))
    ax7.set_ylim(0.0, 1.15); ax7.set_ylabel('Avg test accuracy',fontsize=9)
    ax7.set_title('Test domain summary',fontsize=9)
    ax7.axhline(0.5,color='gray',linestyle='--',alpha=0.4,linewidth=0.8)
    ax7.grid(True,alpha=0.2,axis='y')

    ax8 = fig.add_subplot(2,4,8)
    ax8.axis('off')
    rows = [['','2D (moons)','3D (blobs)'],
            ['Input dim','2','3'],
            ['Dataset','Two moons','Gaussian blobs'],
            ['Drift','Linear θ','Chirp θ(t)'],
            ['Anchor type','20×20 grid','6×6×6 grid'],
            ['Obs. dim','400','216'],
            ['Kalman state','10','10 (same)'],
            ['Adapt method','Identical','Identical']]
    tbl = ax8.table(cellText=rows[1:],colLabels=rows[0],
                    loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1,1.9)
    for j in range(3):
        tbl[(0,j)].set_facecolor('#2171b5')
        tbl[(0,j)].set_text_props(color='white',fontweight='bold')
    for i in [6,7]:
        for j in range(3):
            tbl[(i,j)].set_facecolor('#eaf3de')
    ax8.set_title('2D → 3D scalability',fontsize=9,fontweight='bold')

    plt.tight_layout()
    out = f'{OUT_DIR}/kalman_3d_chirp_results.png'
    plt.savefig(out,dpi=150,bbox_inches='tight')
    print(f"\nSaved → {out}")
    plt.close()


if __name__ == '__main__':
    df, model_list, static_model = run_experiment()
    plot_results(df, model_list, static_model)
    print("\nAll done!")