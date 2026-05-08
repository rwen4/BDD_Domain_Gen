# Recursive Kalman-Based Temporal Adaptation on Rotating 3D Two-Moons Domains

This project investigates whether evolving neural decision boundaries can be modeled as latent temporal signals using anchor-point observations and recursive Kalman filtering.

The benchmark extends the classic Two Moons dataset from 2D to 3D and introduces structured sinusoidal rotational drift over time. Instead of directly retraining on future domains, the framework predicts future decision-boundary evolution from prior temporal observations.

---

## Overview

The core idea is:

1. Generate a sequence of temporally drifting 3D Two-Moons domains
2. Train a neural classifier independently on each observed domain
3. Evaluate each classifier on a fixed set of anchor probes in 3D space
4. Treat the anchor-response vectors as temporal observations
5. Fit a Kalman filter to the evolving anchor trajectories
6. Predict the next domain’s latent decision-boundary state
7. Recursively adapt the classifier using predicted future anchor responses

The experiment evaluates whether temporal structure in evolving decision boundaries can be exploited for future-domain generalization.

---

## Synthetic Drift Environment

The dataset consists of:

- 3D embedded Two Moons manifolds
- Continuous rotational domain drift
- Sinusoidal latent temporal dynamics
- Domain-specific stochastic perturbations

For each domain \(t\), the canonical coordinates \(x_0 \in \mathbb{R}^3\) are transformed as:

$$
x_t = R_z(\theta_t)x_0
$$

where:

$$
R_z(\theta_t)=
\begin{bmatrix}
\cos\theta_t & -\sin\theta_t & 0\\
\sin\theta_t & \cos\theta_t & 0\\
0 & 0 & 1
\end{bmatrix}
$$

The latent rotation angle evolves according to:

$$
\theta_t =
\pi \sin\left(
\frac{2\pi t}{T_{\mathrm{total}}}
\right)
+\epsilon_t
$$

where:
- \(T_{\mathrm{total}}\) is the total number of domains
- \(\epsilon_t\) is stochastic domain noise

This produces:
- continuous nonlinear drift,
- changing rotational direction,
- and nonuniform angular velocity.

---

## Anchor-Based Temporal Representation

Each classifier is evaluated on a fixed set of anchor probes distributed throughout the 3D input space.

For each anchor:
- the classifier outputs:
  
$$
P(y=1 \mid x)
$$

The resulting probability vector becomes the temporal observation used for Kalman filtering.

Unlike explicit geometric tracking, the framework models:
- evolving classifier responses,
- latent boundary motion,
- and temporal decision-boundary dynamics.

---

## Recursive Closed-Loop Kalman Adaptation

The framework operates recursively:

1. Previous domain observations are used to estimate latent temporal dynamics
2. Kalman filtering predicts the next anchor-response state
3. The classifier is adapted toward the predicted future boundary
4. After evaluation, the newly observed domain is incorporated back into the temporal history

This creates an online closed-loop adaptation process capable of tracking evolving domains over time.

---

## Visualization Outputs

The implementation automatically generates:

- Representative drifting 3D domains
- Anchor probability evolution plots
- Aggregate anchor-response trajectories
- Recursive adaptation accuracy curves
- Decision-boundary visualizations

These plots help visualize:
- latent drift structure,
- anchor-response dynamics,
- and recursive temporal adaptation behavior.

---

## Dependencies

Install:

```bash
pip install numpy matplotlib torch scikit-learn pykalman