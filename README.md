# Generator-Coupled Adaptive Source Distribution Learning for Direct Patched Flow Matching DiTs in Multivariate Financial Forecasting

## Version 4.3 (Final Locked Specification & Engineering Implementation Plan)

**Generative core:** Conditional Flow Matching (CFM) operating directly in patched normalized return space, parameterized by a full spatiotemporal Diffusion Transformer (DiT) velocity field and integrated in FP32 via ODE solvers.

**Core research mechanism:** Alternating prior–generator optimization where a context-dependent, conditionally factorized Gaussian source prior $P_\psi(\xi_0 \mid h_t)$ is jointly shaped by strictly proper Gaussian NLL, consistent local distributional transport (debiased Sinkhorn Divergence / Monte Carlo Sliced Wasserstein-2), and a **generator-dependent constant-velocity straightness policy gradient** (unbiased prior to gradient clipping), regularized by a moving-reference per-coordinate KL trust region and separate EMA snapshots.

**Core research question:**

> Can generator-dependent adaptation of a conditional source distribution, combined with context-aware source–target coupling and alternating Flow Matching optimization, reduce the number of ODE evaluations required for calibrated multivariate financial trajectory forecasting without degrading out-of-sample probabilistic quality?

**Three falsifiable research claims:**

1. **Transport efficiency:** Adaptive source priors combined with transport-aware coupling reduce coupling-indexed source-to-target displacement and discrete constant-velocity straightness cost relative to a fixed standard Gaussian source.
2. **Sampling efficiency:** Alternating generator-coupled optimization improves the NFE-versus-forecast-quality Pareto frontier (e.g., matching $32$-NFE Gaussian CFM quality at $8$ NFE) relative to both standard Gaussian CFM and a compute-matched two-stage sequential prior baseline.
3. **Forecast preservation & value added:** These efficiency gains preserve out-of-sample probabilistic calibration, cross-asset dependence, and financial tail realism, while outperforming direct prior-only forecasting (Experiment R) on joint multivariate structure.

---

## Final Lock Changelog (v4.2 $\rightarrow$ v4.3 Final)

All six pre-implementation corrections and early-implementation numerical safeguards are now locked into the specification:

* **Experiment R (Direct Prior-Only Baseline):** Added direct sampling $\hat{Y}_t^{(i)} = \mathcal{P}_p^{-1}(\xi_0^{(i)})$ from $P_\psi(\cdot \mid h_t)$ ($0$ NFE) into Stage 2 and the ablation matrix to isolate the exact multivariate dependence value added by the Flow Matching DiT.
* **Symmetric Mixture-KL Runtime Mask:** Replaced directional $U(i, j)$ gating with the symmetric convexity upper bound $U_{\text{sym}}(i, j) = \frac{1}{2}[U(i, j) + U(j, i)]$ so cross-context compatibility is symmetric.
* **Uniform Sinkhorn Marginals & Sparse-Support Fallback:** Explicitly defined $a = b = [1/B_g, \dots, 1/B_g]^\top$ for guaranteed diagonal feasibility ($\Pi = \operatorname{diag}(1/B_g)$), added dual-potential finiteness and marginal-error checks with automatic fallback to same-context pairing, normalized state/context costs before weighting by $\lambda_h$, and labeled the module explicitly as a **KL-constrained local conditional OT approximation**.
* **Explicit Factored Variogram Score Formula:** Locked down exact pairwise discrepancy $d_p(a, b)$, cardinality normalization across $\mathcal{S}_{\text{temp}}$, $\mathcal{S}_{\text{asset}}$, and $\mathcal{S}_{\text{cross}}$, and fixed validation vs. resampled training pair protocols.
* **Strict Past-Only Volatility Floor:** Moved estimation of $s_{\min, i}$ into a read-only initial calibration prefix ($T_{\text{cal}} = 252$ trading days) prior to the first training origin ($t \ge T_{\text{cal}} + L$) so the zero-leakage perturbation test holds identically on every prediction date.
* **RLOO Unbiasedness Qualification & Numerical Hygiene:** Explicitly qualified that RLOO is unbiased *prior to nonlinear gradient clipping* (`clip_grad_norm_`), added clipping-activation telemetry, enforced FP32 ODE states/updates/accumulators even under BF16 DiT autocast, bumped validation MC mixture-KL estimates to $S = 128$ samples with standard-error reporting, and integrated a complete **Implementation Plan** (Section 11).

---

# 1. Problem Definition, Direct Configurable Patching, and Notation

## 1.1 Multivariate Forecasting Task

For $N$ equities at trading date $t$, the model observes a historical window of $L$ trading days and generates an ensemble of $M$ joint future return trajectories over the next $H$ trading days:

$$
X_t^{\text{past}} \in \mathbb{R}^{L \times N \times F}, \qquad Y_t^{\text{future}} \in \mathbb{R}^{H \times N \times 1}, \qquad \left\{\hat{Y}_t^{(1)}, \dots, \hat{Y}_t^{(M)}\right\} \subset \mathbb{R}^{H \times N \times 1}.
$$

Default initial dimensions are $L = 60$, $H = 20$, $N = 50$, and $F = 12$ compact historical features per asset.

## 1.2 Lossless Direct Temporal Patching Operator (No VAE)

Because $Y_t^{\text{future}}$ contains only $H \cdot N \cdot 1 = 1,000$ scalar values, a latent autoencoder is unnecessary and introduces reconstruction bias and posterior-collapse risk. Define a lossless temporal patching operator $\mathcal{P}_p$ with configurable patch length $p \in \{1, 2, 4\}$ (where $H_p = H / p \in \mathbb{Z}^+$):

$$
Y_t^{\text{patch}} = \mathcal{P}_p(Y_t^{\text{future}}) \in \mathbb{R}^{H_p \times N \times p},
$$

where token $(k, i)$ for temporal patch $k \in \{0, \dots, H_p - 1\}$ and asset $i \in \{1, \dots, N\}$ is:

$$
Y_{t, k, i}^{\text{patch}} = \begin{bmatrix} \tilde{r}_{i, t + pk + 1} & \tilde{r}_{i, t + pk + 2} & \cdots & \tilde{r}_{i, t + pk + p} \end{bmatrix}^\top \in \mathbb{R}^p.
$$

The inverse unpatchification operator $\mathcal{P}_p^{-1}: \mathbb{R}^{H_p \times N \times p} \to \mathbb{R}^{H \times N \times 1}$ is a zero-parameter reshape/transpose. Across all patch sizes, the total flow state dimensionality stays identically $D = H_p \cdot N \cdot p = 1,000$:

| Patch Size $p$ | Temporal Patches $H_p$ | Total DiT Tokens $T_{\text{tok}} = H_p N$ | Attention Matrix Entries | Flow State Dim $D$ | Role |
|---|---|---|---|---|---|
| $p = 4$ | $5$ | $250$ | $62,500$ | $1,000$ | Default starting point (fastest) |
| $p = 2$ | $10$ | $500$ | $250,000$ | $1,000$ | Primary granularity ablation |
| $p = 1$ | $20$ | $1,000$ | $1,000,000$ | $1,000$ | Full daily-token upper bound |

## 1.3 Complete Notation Table

| Symbol | Shape / Domain | Definition |
|---|---|---|
| $X_t$ | $\mathbb{R}^{L \times N \times F}$ | Historical input window ending at date $t$ |
| $Y_t^{\text{patch}}$ | $\mathbb{R}^{H_p \times N \times p}$ | Ground-truth patched future normalized log returns ($D = 1,000$) |
| $h_t$ | $(h_t^{\text{stocks}}, h_t^{\text{global}})$ | Encoded historical market context: $\mathbb{R}^{N \times d_h}$ and $\mathbb{R}^{d_h}$ |
| $\xi_0$ | $\mathbb{R}^{H_p \times N \times p}$ | Initial source sample drawn from $P_\psi(\cdot \mid h_t)$ |
| $z_\tau$ | $\mathbb{R}^{H_p \times N \times p}$ | Interpolated flow state at continuous time $\tau \in [0, 1]$ |
| $z(1)$ | $\mathbb{R}^{H_p \times N \times p}$ | Terminal ODE state at $\tau = 1$ (reserving $K$ strictly for ODE step count) |
| $u^*$ | $\mathbb{R}^{H_p \times N \times p}$ | Conditional linear target velocity $Y_t^{\text{patch}} - \xi_0$ |
| $E_\phi$ | Module | Historical market encoder (Temporal Transformer + Signed-Edge GATv2 + AttentionPool) |
| $P_\psi, P_{\bar{\psi}}$ | Module | Online and EMA conditionally factorized Gaussian source priors |
| $v_\theta, v_{\bar{\theta}}$ | Module | Online and EMA Flow Matching DiT velocity fields (includes output projection) |
| $G_\theta$ | Operator | Complete FP32 ODE integration $\xi_0 \mapsto z(1)$ followed by unpatchification $\mathcal{P}_p^{-1}$ |
| $P_{\text{ref}}^{(b)}$ | Frozen snapshot | Previous outer iteration's EMA prior $P_{\bar{\psi}^{(b)}}$ |
| $q_b$ | Frozen mixture | Block B source mixture $\rho P_{\text{new}}^{(b)} + (1 - \rho) P_{\text{old}}^{(b)}$ |
| $b$ | $\mathbb{Z}_{\ge 0}$ | Outer alternating optimization iteration |
| $K$ | $\mathbb{Z}^+$ | Number of fixed-step ODE solver steps |
| $M$ | $\mathbb{Z}^+$ | Candidate rollouts per context ($M \in \{20, 64, 128\}$; even when using antithetic pairs) |
| $\operatorname{sg}(\cdot)$ | Operator | Stop-gradient (`.detach()`), blocking backpropagation through its argument |

---

# 2. Financial Data Pipeline, Causal Volatility Floor, and Leakage Controls

## 2.1 Universe and Compact Feature Set

Start with $N = 30\text{--}50$ liquid equities (e.g., Indian equities via NSE historical archives / adjusted daily feeds). Use a compact $F = 12$ feature vector per asset per day:

1. Daily adjusted log return $r_{i,t}$ and normalized return $\tilde{r}_{i,t}$.
2. Log high–low range $\log(H_{i,t}/L_{i,t})$ and open–close return $\log(C_{i,t}/O_{i,t})$.
3. Log volume change $\log((V_{i,t} + 1)/(V_{i,t-1} + 1))$.
4. Past-only rolling volatility $s_{i,t}$ ($20$-day) and downside semi-volatility ($20$-day).
5. Multi-horizon momentum: $5$-day, $20$-day, and $60$-day cumulative log returns.
6. Broad market index log return and sector index log return.

## 2.2 Strictly Causal Volatility Floor & Past-Only Scaling

Compute adjusted daily log returns:

$$
r_{i,t} = \log\left(\frac{P_{i,t}^{\text{adj}}}{P_{i,t-1}^{\text{adj}}}\right).
$$

**Calibration-prefix floor (Zero-Leakage Resolution):** Reserve the first $T_{\text{cal}} = 252$ trading days of the dataset strictly as a read-only calibration prefix. Compute each asset's volatility floor $s_{\min, i}$ as the $5\text{th}$ percentile of its $20$-day rolling volatility over $t \in [W, T_{\text{cal}}]$ and freeze $s_{\min}$ permanently. Begin all training prediction origins strictly at $t \ge T_{\text{cal}} + L$.

For every valid prediction origin $t$, compute the strictly causal rolling volatility scale over window $W = 20$ trading days ending at $t$:

$$
s_{i,t} = \max\left(s_{\min, i},\; \sqrt{\frac{1}{W - 1}\sum_{u = t - W + 1}^{t}(r_{i,u} - \bar{r}_{i,t})^2 + \varepsilon}\right).
$$

Normalize historical returns and future targets using only information available at or before date $t$:

$$
\tilde{r}_{i,t} = \frac{r_{i,t}}{s_{i,t}}, \qquad \tilde{r}_{i, t + h} = \frac{r_{i, t + h}}{s_{i,t}} \quad (h = 1, \dots, H).
$$

Do **not** clip raw targets in evaluation datasets; always evaluate on unclipped raw log returns and reconstructed price paths.

## 2.3 Chronological Splits, Purge Gaps, and Overlap Stride Controls

1. **Strict chronological partitions:** Calibration prefix ($252$ days) $\rightarrow$ Train (e.g., 2012--2020) $\rightarrow$ Validation (2021--2022) $\rightarrow$ Test (2023--2025).
2. **Boundary purge:** Drop $25$ trading days ($> H = 20$) immediately after the last training prediction date and after the last validation prediction date so zero target days cross a partition boundary.
3. **Automated zero-leakage unit test:** Perturbing all market observations strictly after date $t$ must change $X_t$, $s_{i,t}$, and the historical graph adjacency matrix $A_t$ by identically `0.0` for every prediction origin $t$ across train, validation, and test.
4. **Target overlap reduction in Block A:** Because a $20$-day horizon at stride $S = 1$ shares $95\%$ of target days across adjacent windows, Block A samples windows using stride $S \in \{5, 10, 20\}$ with a randomized epoch starting offset $o \in \{0, \dots, S - 1\}$. Stride $S = 5$ reduces overlap to $(20 - 5)/20 = 75\%$; non-overlapping targets occur at $S = 20$. Compare $S \in \{5, 10, 20\}$ empirically.
5. **Survivorship control:** Use point-in-time historical index constituents where available; otherwise explicitly document fixed-universe selection bias.

---

# 3. Historical Market Representation Encoder ($E_\phi$)

The historical encoder $E_\phi$ maps $X_t \in \mathbb{R}^{L \times N \times F}$ into structured stock-level and global conditioning tokens.

## 3.1 Three-Stage Architecture

1. **Per-Asset Temporal Transformer:** Project each asset's history $X_{t, i} \in \mathbb{R}^{L \times F}$ to width $d_h = 128$, add learned temporal positional embeddings, apply $L_{\text{temp}} = 2\text{--}3$ Transformer encoder layers, and attention-pool over the $L$ days to obtain $u_{t,i} \in \mathbb{R}^{d_h}$.
2. **Cross-Stock Signed-Edge Shrinkage Graph Encoder:** Construct a strictly past-only $N \times N$ correlation matrix $A_t$ from the $60$-day lookback returns using Ledoit–Wolf shrinkage. Represent each retained top-$k$ edge $(i, j)$ using a **signed edge feature vector** $e_{ij}^{\text{edge}} = \big[\max(A_{t,ij}, 0),\; \max(-A_{t,ij}, 0),\; |A_{t,ij}|\big] \in \mathbb{R}^3$ fed into $L_{\text{gnn}} = 2$ edge-conditioned GATv2 layers over $\{u_{t,i}\}_{i=1}^N$ to obtain relational embeddings $v_{t,i} \in \mathbb{R}^{d_h}$. Monitor edge sparsity and correlation-graph stability across market regimes.
3. **Global Market-Context Pooling & Fusion:** Aggregate across all $N$ assets via multi-head attention pooling:

$$
h_t^{\text{global}} = \operatorname{MLP}_\phi\big(\operatorname{AttentionPool}(v_{t,1}, \dots, v_{t,N})\big) \in \mathbb{R}^{d_h},
$$

and fuse temporal, relational, and stock/sector identity embeddings into stock context tokens:

$$
h_{t, i}^{\text{stocks}} = \operatorname{LayerNorm}\big(u_{t,i} + v_{t,i} + e_i^{\text{stock}} + e_{\text{sec}(i)}^{\text{sector}}\big) \in \mathbb{R}^{d_h}.
$$

The encoder outputs $h_t = \big(h_t^{\text{stocks}} \in \mathbb{R}^{N \times d_h},\; h_t^{\text{global}} \in \mathbb{R}^{d_h}\big)$.

## 3.2 Training and Freezing Protocol

Train $E_\phi$ jointly with the baseline Gaussian-source CFM DiT in Stage 1 (optionally warm-started with masked historical feature reconstruction). Once Stage 1 converges, **freeze $E_\phi$ permanently** across all prior-adaptation and alternating iterations so context neighborhoods, KL divergences, and transport distances remain anchored to a stationary representation space.

---

# 4. Conditionally Factorized Gaussian Source Prior ($P_\psi$)

The source prior defines a context-dependent starting distribution directly in patched return space $\mathbb{R}^{H_p \times N \times p}$:

$$
P_\psi(\xi_0 \mid h_t) = \mathcal{N}\!\left(\mu_\psi(h_t),\; \operatorname{diag}\!\big(\sigma_\psi^2(h_t)\big)\right), \qquad \mu_\psi(h_t),\, \sigma_\psi(h_t) \in \mathbb{R}^{H_p \times N \times p}.
$$

## 4.1 Explicit Architectural Role and Covariance Scope

This baseline prior is a **conditionally factorized Gaussian**: conditioned on the full market context $h_t$, it predicts marginal drift $\mu_\psi(h_t)$ and marginal scale $\sigma_\psi(h_t)$ for every coordinate while treating stochastic draws across coordinates as conditionally independent.

* **Prior responsibility:** Capture context-dependent marginal location and scale shifts so the flow does not spend capacity rescaling isotropic $\mathcal{N}(0, I)$ noise.
* **DiT responsibility:** Learn the complete joint multivariate dependence structure—cross-stock correlations, sector co-movements, skewness, volatility clustering, and fat tails.

**Stage 6 covariance extension:** After establishing the diagonal baseline, compare against a low-rank factor Gaussian prior $\Sigma_\psi(h_t) = D_\psi(h_t) + B_\psi(h_t)B_\psi(h_t)^\top$ with factor rank $r \in \{8, 16\}$ (capturing shared market/sector shocks before the ODE starts) and a true conditional Student-$t$ prior.

## 4.2 Dual-Head Architecture, Exact $\mathcal{N}(0, I)$ Initialization & Bounded Scales

Condition each stock token $h_{t,i}^{\text{stocks}}$ on $h_t^{\text{global}}$ via FiLM/AdaLN and pass through a shared lightweight per-stock MLP trunk producing hidden features $f_{t,i} \in \mathbb{R}^{d_{\text{prior}}}$. Use **two separate linear output heads** per stock:

```python
self.mu_head = nn.Linear(d_prior, H_p * p)
self.scale_head = nn.Linear(d_prior, H_p * p)

b0 = math.log((1.0 - sigma_min) / (sigma_max - 1.0))  # ~ -1.3218 for (0.20, 4.00)

nn.init.zeros_(self.mu_head.weight)
nn.init.zeros_(self.mu_head.bias)

nn.init.zeros_(self.scale_head.weight)
nn.init.constant_(self.scale_head.bias, b0)
```

Reshape across stocks into $\mu_\psi(h_t), s_\psi(h_t) \in \mathbb{R}^{H_p \times N \times p}$ and enforce strict positive bounds analytically by construction:

$$
\sigma_\psi(h_t) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \operatorname{sigmoid}\!\big(s_\psi(h_t)\big),
$$

with default bounds $\sigma_{\min} = 0.20$ and $\sigma_{\max} = 4.00$. At initialization, $\mu_\psi(h_t) \equiv 0$ and $\sigma_\psi(h_t) \equiv 1$ identically ($\mathcal{N}(0, I)$).

**Boundary-saturation diagnostic:** Continuously log the fraction of coordinates with $\sigma_{\psi, d} < \sigma_{\min} + 0.02$ or $\sigma_{\psi, d} > \sigma_{\max} - 0.05$. If saturation exceeds $5\%$ on validation, widen bounds or evaluate a heavy-tailed prior family.

## 4.3 Strict Gradient-Ownership Rules for Sampling

Reparameterized sampling is $\xi_0 = \mu_\psi(h_t) + \sigma_\psi(h_t) \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$:

* **Pathwise distributional transport ($\mathcal{L}_{\text{SD}}$ or $\mathcal{L}_{\text{SW2}}$):** Use attached `rsample` so gradients flow through $\xi_0$ into $(\mu_\psi, \sigma_\psi)$.
* **Straightness policy gradient ($\mathcal{L}_{\text{straight-PG}}$):** Explicitly **detach** $\xi_0$ (`xi_0.detach()`) before evaluating `log_prob_sum` to prevent Gaussian score cancellation.
* **Generator Block B:** Sample from frozen prior snapshots under `torch.no_grad()` / `.detach()`.

---

# 5. Direct Patched Conditional Flow Matching DiT ($v_\theta$)

## 5.1 Tokenization and Full Spatiotemporal DiT Backbone

Given an interpolated patched return tensor $z_\tau \in \mathbb{R}^{H_p \times N \times p}$, continuous flow time $\tau \in [0, 1]$, and context $h_t$:

1. **Patch input projection:** Project each $p$-day return patch $x_{k,i} \in \mathbb{R}^p$ into internal Transformer width $d_{\text{model}} = 128$:

$$
e_{k,i}^{(0)} = W_{\text{in}} x_{k,i} + b_{\text{in}} + e_k^{\text{patch}} + e_i^{\text{stock}} + e_{\text{sec}(i)}^{\text{sector}} \in \mathbb{R}^{d_{\text{model}}}.
$$

Flatten across $(H_p, N)$ into a sequence of $T_{\text{tok}} = H_p N$ tokens ($250$ tokens for $p = 4$).

2. **DiT blocks ($L_{\text{DiT}} = 6\text{--}8$ layers):** Each block applies **full joint 2D spatiotemporal self-attention**, cross-attention to $h_t^{\text{stocks}} \in \mathbb{R}^{N \times d_h}$, and a SwiGLU FFN. All sublayers are modulated by **AdaLN-Zero** conditioned on $c_\tau = \operatorname{MLP}(\operatorname{SinusoidalEmb}(\tau)) + \operatorname{Proj}(h_t^{\text{global}})$.
3. **Output velocity projection:** Apply final AdaLN and a zero-initialized linear head $W_{\text{out}} \in \mathbb{R}^{p \times d_{\text{model}}}$, reshaping back to:

$$
v_\theta(z_\tau, \tau, h_t) \in \mathbb{R}^{H_p \times N \times p}.
$$

## 5.2 Linear Probability Path, Flow-Time Sampling & Canonical MSE Objective

For a paired source sample $\xi_0$ and patched target $Y_t^{\text{patch}}$:

$$
z_\tau = (1 - \tau)\xi_0 + \tau Y_t^{\text{patch}}, \qquad u^* = \frac{dz_\tau}{d\tau} = Y_t^{\text{patch}} - \xi_0, \qquad \tau \in [0, 1].
$$

* **Canonical flow-time distribution:** $\tau \sim \mathcal{U}(0, 1)$ (Logit-Normal $\tau = \operatorname{sigmoid}(s),\, s \sim \mathcal{N}(0, 1)$ isolated in Exp M2).
* **Canonical CFM objective (MSE):**

$$
\mathcal{L}_{\text{CFM}}^{(b)} = \mathbb{E}_{(\xi_0, Y^{\text{patch}}, h_j) \sim \pi_b,\; \tau \sim p_\tau}\!\left[\frac{1}{D}\left\|v_\theta(z_\tau, \tau, h_j) - \big(Y^{\text{patch}} - \xi_0\big)\right\|_2^2\right].
$$

MSE remains primary because its exact population minimizer is the conditional expectation $v^*(z_\tau, \tau, h) = \mathbb{E}[u^* \mid z_\tau, \tau, h]$. Keep Pseudo-Huber CFM and tail-weighted MSE strictly as separate ablations (Exp P).

## 5.3 Source–Target Coupling: Independent Baseline vs. KL-Constrained Local Conditional OT Approximation

### Why Coupling Is a First-Class Component

Even when $P_\psi = p_{\text{data}} = \mathcal{N}(0, I)$, independent linear interpolation $z_\tau = (1 - \tau)\xi_0 + \tau Y$ yields intermediate variance $\operatorname{Var}(z_\tau) = (1 - \tau)^2 + \tau^2$, contracting to $\operatorname{Var}(z_{0.5}) = 0.5$ before re-expanding—forcing a nontrivial transport map. Optimal identity coupling ($\xi_0 = Y$) instead yields $u^* = 0$. Therefore, v4.3 elevates the comparison between **Same-Context Independent Coupling** and **KL-Constrained Local Conditional OT Approximation** into a primary study (Stage 3 & Exp L), and requires every reported transport displacement to state its exact coupling $\pi$:

$$
D_{\text{transport}}(\pi) = \mathbb{E}_{(\xi_0, Y^{\text{patch}}) \sim \pi}\!\left[\frac{1}{D}\left\|Y^{\text{patch}} - \xi_0\right\|_2^2\right].
$$

### KL-Constrained Local Conditional OT Approximation (Uniform Marginals + Symmetric Mask)

Retrieve local neighborhoods $g$ of size $B_g \in \{16, 32\}$ in normalized frozen global context space $\hat{h}^{\text{global}}$ (optionally requiring matching coarse volatility regime $r_i = r_j$). Because financial history gives one realized trajectory per date, this is explicitly a **local conditional OT approximation**, not exact conditional OT.

1. **Uniform Matching Marginals (Guaranteed Diagonal Feasibility):** Fix uniform source and target weights over the indexed neighborhood minibatch:

$$
a = b = \left[\frac{1}{B_g}, \dots, \frac{1}{B_g}\right]^\top \in \Delta^{B_g - 1}.
$$

2. **Symmetric Convexity Upper-Bound Runtime Mask:** Because $q_b = \rho P_{\text{new}} + (1 - \rho)P_{\text{old}}$ is a Gaussian mixture with no closed-form KL, define the directional convexity upper bound:

$$
U(i, j) = \rho\,\overline{D}_{KL}\!\big(P_{\text{new}}(\cdot \mid h_i) \Vert P_{\text{new}}(\cdot \mid h_j)\big) + (1 - \rho)\,\overline{D}_{KL}\!\big(P_{\text{old}}(\cdot \mid h_i) \Vert P_{\text{old}}(\cdot \mid h_j)\big),
$$

symmetrize it for neighborhood compatibility, and construct the binary mask:

$$
U_{\text{sym}}(i, j) = \frac{1}{2}\big(U(i, j) + U(j, i)\big), \qquad M_{ij} = \mathbf{1}\!\left[U_{\text{sym}}(i, j) \le \delta_{\text{context}} \;\land\; r_i = r_j\right].
$$

Because $U_{\text{sym}}(i, i) = 0 \le \delta_{\text{context}}$ and $a = b = \mathbf{1}/B_g$, the diagonal plan $\Pi = \operatorname{diag}(1/B_g, \dots, 1/B_g)$ is **provably feasible** in the masked polytope.

3. **Scale-Normalized Cost Matrix & Explicitly Masked Log-Domain Sinkhorn:** Normalize both state squared distances $C_{ij}^{\text{state}} = \frac{1}{D}\|\xi_i - Y_j^{\text{patch}}\|_2^2$ and context distances $C_{ij}^{\text{ctx}} = \|\hat{h}_i^{\text{global}} - \hat{h}_j^{\text{global}}\|_2^2$ by their respective neighborhood medians ($\tilde{C}^{\text{state}}, \tilde{C}^{\text{ctx}}$) so $\lambda_h$ is dimensionless and tunable. Set infeasible off-diagonal log-kernel entries explicitly to $-\infty$ (never finite $10^6$ penalties):

$$
\log K_{ij} = \begin{cases} -\dfrac{1}{\varepsilon_{\text{OT}}}\left(\tilde{C}_{ij}^{\text{state}} + \lambda_h \tilde{C}_{ij}^{\text{ctx}}\right), & \text{if } M_{ij} = 1, \\[10pt] -\infty, & \text{if } M_{ij} = 0. \end{cases}
$$

4. **Convergence Check & Automatic Fallback:** Run stabilized masked log-domain Sinkhorn iterations in FP32. Verify that dual potentials are finite and marginal error satisfies $\|\Pi\mathbf{1} - a\|_1 + \|\Pi^\top\mathbf{1} - b\|_1 < 10^{-4}$. If sparse support causes convergence failure, automatically fall back to $\Pi = \operatorname{diag}(1/B_g)$ (same-context pairing) and log the fallback rate. Always condition the DiT on **`paired_context = h_j`**.

5. **Monte Carlo Estimate of Symmetric Mixture KL (Validation Diagnostic):** On validation neighborhoods, compute an $S = 128$-sample Monte Carlo estimate (plus standard error across repeated draws) of true symmetric mixture KL using full joint summed log-densities inside `logsumexp`, dividing by $D$ only after taking the difference:

$$
\log q_b(\xi \mid h) = \operatorname{logsumexp}\!\begin{bmatrix} \log \rho + \log P_{\text{new}}^{\text{sum}}(\xi \mid h) \\ \log(1 - \rho) + \log P_{\text{old}}^{\text{sum}}(\xi \mid h) \end{bmatrix},
$$

$$
\widehat{\overline{D}}_{KL}^{\text{MC}}(q_i \Vert q_j) = \frac{1}{S D}\sum_{s=1}^{S}\Big[\log q_b(\xi_s \mid h_i) - \log q_b(\xi_s \mid h_j)\Big], \qquad \widehat{\overline{D}}_{\text{sym}}^{\text{MC}} = \frac{1}{2}\Big(\widehat{\overline{D}}_{KL}^{\text{MC}}(q_i \Vert q_j) + \widehat{\overline{D}}_{KL}^{\text{MC}}(q_j \Vert q_i)\Big).
$$

## 5.4 Pushforward-Invariance Principle

Under infinite model capacity and exact ODE integration, minimizing $\mathcal{L}_{\text{CFM}}^{(b)}$ transports any full-support source distribution $q_b(\cdot \mid h_t)$ to the identical target distribution $p(Y_t^{\text{patch}} \mid h_t)$. Therefore, v4.3 adapts $P_\psi$ to reduce transport displacement, enforce constant-velocity straightness, and cut inference NFE while preserving unbiased probabilistic calibration.

---

# 6. FP32 ODE Integration & Discrete Constant-Velocity Straightness Cost

## 6.1 FP32 Fixed-Step Heun Integrator

Even when `PatchedFlowDiT` executes internally under `bfloat16` autocast, **all ODE states, predictor states, and velocity updates remain strictly in `float32`** to prevent low-precision drift across integration steps. Given $\xi_0 \sim P_{\text{source}}(\cdot \mid h_t)$ and $K$ steps ($\Delta\tau = 1/K$, $\tau_m = m\Delta\tau$):

$$
z_0 = \operatorname{fp32}(\xi_0),
$$

$$
k_{1,m} = \operatorname{fp32}\!\big(v_\theta(z_m, \tau_m, h_t)\big), \qquad z_{\text{pred}, m} = z_m + \Delta\tau\, k_{1,m},
$$

$$
k_{2,m} = \operatorname{fp32}\!\big(v_\theta(z_{\text{pred}, m}, \tau_{m+1}, h_t)\big), \qquad z_{m+1} = z_m + \frac{\Delta\tau}{2}(k_{1,m} + k_{2,m}).
$$

Each step costs $2$ NFE ($2K$ total NFE). Denote the terminal ODE state as $z(1) \triangleq z_K$, and unpatchify to $\hat{Y}_t = \mathcal{P}_p^{-1}(z(1))$.

## 6.2 Discrete Constant-Velocity Straightness Cost

Define the continuous **Constant-Velocity Straightness Cost** along path $z(\tau)$ with total displacement $\Delta z = z(1) - z(0)$:

$$
C_{\text{straight}}(\xi_0; h_t) = \frac{1}{D}\int_0^1 \left\|v(z(\tau), \tau, h_t) - \Delta z\right\|_2^2 d\tau.
$$

This measures deviation from constant-velocity straight-line motion (not purely geometric curvature): when $C_{\text{straight}} = 0$, a single Euler step recovers $z(1) = \xi_0 + \Delta z$ identically regardless of $\|\Delta z\|_2$, preserving tail trajectories.

Under $K$-step Heun integration, $\Delta z = \frac{1}{2K}\sum_{m=0}^{K-1}(k_{1,m} + k_{2,m})$. By the variance identity, the discrete Heun trapezoidal straightness estimator simplifies **exactly** (for the discrete $K$-step Heun grid) to:

$$
\hat{C}_{\text{straight}}^{(K)}(\xi_0; h_t) = \frac{1}{D}\left[\frac{1}{2K}\sum_{m=0}^{K-1}\left(\|k_{1,m}\|_2^2 + \|k_{2,m}\|_2^2\right) - \|z(1) - z(0)\|_2^2\right].
$$

**Engineering & diagnostic properties:**

* **Zero extra NFE over an existing Heun rollout**, though evaluating $M$ rollouts in Block A still requires $2KM$ DiT evaluations per context ($320$ calls/context at $M = 20, K = 8$), tracked explicitly in total compute budgets.
* **$O(D)$ working memory, independent of $K$:** Stores only current FP32 $z$, $z_{\text{pred}}$, $k_1$, $k_2$, and a scalar FP32 accumulator (clamped at `0.0`).
* **Deterministic evaluation:** Frozen EMA DiT is always placed in `.eval()` mode during Block A rollouts.
* **Multi-resolution check:** Log Spearman rank correlation of $\hat{C}_{\text{straight}}^{(K)}$ across $K \in \{4, 8, 16, 32\}$ to confirm the prior is not exploiting coarse-grid artifacts.

---

# 7. Block A: Generator-Coupled Prior Optimization

At the start of outer iteration $b$, freeze immutable snapshots $P_{\text{ref}}^{(b)} = P_{\bar{\psi}^{(b)}}$ and $v_{\text{ref}}^{(b)} = v_{\bar{\theta}^{(b)}}$ (both in `.eval()` mode). Only online prior parameters $\psi$ are trainable.

## 7.1 Strictly Proper Coordinate-Normalized Gaussian NLL

$$
\mathcal{L}_{\text{NLL}}(\psi) = -\frac{1}{BD}\sum_{t=1}^{B}\log P_\psi^{\text{sum}}\!\big(Y_t^{\text{patch}} \mid h_t\big) = \frac{1}{2BD}\sum_{t=1}^{B}\sum_{d=1}^{D}\left[\log\!\big(2\pi\sigma_{\psi,t,d}^2\big) + \frac{(Y_{t,d}^{\text{patch}} - \mu_{\psi,t,d})^2}{\sigma_{\psi,t,d}^2}\right].
$$

Never mix a Student-$t$ log-likelihood with a Gaussian sampling prior. Rely on bounded $\sigma_{\max}$, gradient clipping, and boundary-saturation logging; defer a true conditional Student-$t$ prior (with Student-$t$ sampling and MC KL) to Stage 6.

## 7.2 Consistent Local Distributional Transport: Sinkhorn Divergence & Monte Carlo $\widehat{\text{SW}}_2^2$

1. **Form tight local context neighborhoods first (Approach A):** Group training windows into local neighborhoods $g \in G$ of size $B_g \in \{24, 32\}$ via nearest neighbors in normalized frozen $\hat{h}_t^{\text{global}}$ (optionally requiring regime agreement). Draw **attached reparameterized samples** $\xi_i = \mu_\psi(h_i) + \sigma_\psi(h_i) \odot \epsilon_i$ (`rsample`) for $i \in g$, and collect detached realized targets $Q_g = \{Y_j^{\text{patch}}\}_{j \in g}$.
2. **Consistent Local Sinkhorn Divergence:** Apply the **exact same pure state ground cost** $c(x, y) = \frac{1}{D}\|x - y\|_2^2$ across all three entropic OT terms:

$$
S_\varepsilon(P_{\psi, g}, Q_g) = \operatorname{OT}_\varepsilon(P_{\psi, g}, Q_g) - \frac{1}{2}\operatorname{OT}_\varepsilon(P_{\psi, g}, P_{\psi, g}) - \frac{1}{2}\operatorname{OT}_\varepsilon(Q_g, Q_g).
$$

3. **Monte Carlo Approximation of Empirical Sliced Wasserstein-2 ($\widehat{\text{SW}}_2^2$):** Over $L_{\text{proj}} \in \{128, 256, 512\}$ random unit directions $\theta_\ell \sim \mathcal{U}(\mathbb{S}^{D-1})$ (evaluated across multiple projection seeds for Monte Carlo variance diagnostics):

$$
\mathcal{L}_{\text{SW2}}(\psi) = \frac{1}{|G| L_{\text{proj}} B_g}\sum_{g \in G}\sum_{\ell = 1}^{L_{\text{proj}}}\sum_{r = 1}^{B_g}\left(\operatorname{sort}\!\big(\{\theta_\ell^\top \xi_i\}_{i \in g}\big)_{(r)} - \operatorname{sort}\!\big(\{\theta_\ell^\top Y_j^{\text{patch}}\}_{j \in g}\big)_{(r)}\right)^2,
$$

with per-group complexity $O\big(L_{\text{proj}} B_g (D + \log B_g)\big)$ and zero entropic bias. Compare local Sinkhorn Divergence, Sliced Sinkhorn, and $\widehat{\text{SW}}_2^2$ in Stage 2 to select the default $\mathcal{L}_{\text{dist}}$.

## 7.3 Generator-Dependent Straightness Policy Gradient (RLOO)

All candidate samples $\xi_{0,t}^{(i)}$ are rolled out under `torch.no_grad()` in FP32 through frozen `dit_ref.eval()` to obtain $C_{t,i} = \hat{C}_{\text{straight}}^{(K)}(\xi_{0,t}^{(i)}; h_t)$ (already divided by $D$), and explicitly **detached** before calling `log_prob_sum`.

* **Canonical Baseline — Independent-Sample RLOO:** Draw $M$ i.i.d. samples $\xi_{0,t}^{(i)} = \operatorname{sg}(\mu_\psi(h_t) + \sigma_\psi(h_t) \odot \epsilon_{t,i})$:

$$
b_{t, -i} = \frac{1}{M - 1}\sum_{j \neq i} C_{t, j}, \qquad A_{t, i} = \frac{C_{t, i} - b_{t, -i}}{\hat{\sigma}_{C, \text{prev}} + \varepsilon}.
$$

* **Ablation Variant — Pair-Level Antithetic RLOO (Exp M1):** Draw $M_p = M/2$ pairs $\xi_{t,m}^\pm = \operatorname{sg}(\mu_\psi \pm \sigma_\psi \odot \epsilon_{t,m})$, compute pair means $\bar{C}_{t,m} = \frac{1}{2}(C_{t,m}^+ + C_{t,m}^-)$, and exclude the entire pair $m$:

$$
b_{t, -m} = \frac{1}{M_p - 1}\sum_{n \neq m} \bar{C}_{t, n}, \qquad A_{t, m}^\pm = \frac{C_{t, m}^\pm - b_{t, -m}}{\hat{\sigma}_{C, \text{prev}} + \varepsilon}.
$$

* **Full Joint Summed Log-Probability & Past-Batch Scale:**

$$
\mathcal{L}_{\text{straight-PG}}(\psi) = \frac{1}{B M}\sum_{t=1}^{B}\sum_{i=1}^{M} \operatorname{sg}\!\big(A_{t, i}\big)\,\log P_\psi^{\text{sum}}\!\Big(\operatorname{sg}\!\big(\xi_{0, t}^{(i)}\big) \mid h_t\Big).
$$

* **Strict Estimator Qualifications:**
  1. Read $\hat{\sigma}_{C, \text{prev}}$ from **strictly previous batches** *before* calling `running_cost_std.update(costs)`.
  2. Leave $A_{t,i}$ unclipped in the canonical mode (advantage clipping is a labeled biased ablation).
  3. **Unbiasedness vs. gradient clipping:** Both score-function gradient estimators are mathematically unbiased **prior to nonlinear gradient clipping** (`clip_grad_norm_`). Because clipping introduces bias into the practical optimizer step whenever activated, log the **clipping activation percentage** every epoch alongside the live gradient-norm ratio $\|\nabla_\psi(\lambda_{\text{straight}}\mathcal{L}_{\text{straight-PG}})\|_2 / \|\nabla_\psi(\lambda_{\text{NLL}}\mathcal{L}_{\text{NLL}})\|_2$.

## 7.4 Moving-Reference Per-Coordinate KL Trust Region & Epoch Dual Update

$$
\overline{D}_{KL}(P_\psi \Vert P_{\text{ref}}^{(b)}) = \frac{1}{2 B D}\sum_{t=1}^{B}\sum_{d=1}^{D}\left[\log\frac{\sigma_{r,t,d}^2}{\sigma_{\psi,t,d}^2} + \frac{\sigma_{\psi,t,d}^2 + (\mu_{\psi,t,d} - \mu_{r,t,d})^2}{\sigma_{r,t,d}^2} - 1\right].
$$

The primal Block A objective is:

$$
\mathcal{L}_P^{(b)}(\psi) = \lambda_{\text{NLL}}\mathcal{L}_{\text{NLL}}(\psi) + \lambda_{\text{dist}}\mathcal{L}_{\text{dist}}(\psi) + \lambda_{\text{straight}}\mathcal{L}_{\text{straight-PG}}(\psi) + \beta_b\, \overline{D}_{KL}\!\big(P_\psi \Vert P_{\text{ref}}^{(b)}\big).
$$

Update $\beta_b$ **once per epoch** using epoch-mean detached KL $\widehat{D}_{KL}^{\text{epoch}}$:

$$
\beta_b \leftarrow \operatorname{clip}\!\left(\beta_b \exp\!\Big[\eta_\beta\big(\widehat{D}_{KL}^{\text{epoch}} - \delta\big)\Big],\; \beta_{\min},\; \beta_{\max}\right),
$$

with $\delta \in [0.02, 0.05]$ nats/coord, $\beta_0 = 1.0$, $\eta_\beta = 0.5$, $\beta_{\min} = 0.05$, $\beta_{\max} = 50.0$. Update `ema_prior` ($\bar{\psi} \leftarrow \alpha_P \bar{\psi} + (1 - \alpha_P)\psi$) after every step while keeping $P_{\text{ref}}^{(b)}$ frozen.

---

# 8. Block B: Source-Matched Generator Optimization & Explicit Scoring Rules

Freeze $P_{\text{old}}^{(b)} = P_{\bar{\psi}^{(b)}}$ and $P_{\text{new}}^{(b)} = P_{\bar{\psi}^{(b+1)}}$, and form the fixed Block B mixture $q_b(\xi_0 \mid h_t) = \rho P_{\text{new}}^{(b)}(\xi_0 \mid h_t) + (1 - \rho) P_{\text{old}}^{(b)}(\xi_0 \mid h_t)$ ($\rho = 0.70$). Only online DiT parameters $\theta$ are trainable:

$$
\mathcal{L}_G^{(b)}(\theta) = \lambda_{\text{CFM}}\mathcal{L}_{\text{CFM}}^{(b)}(\theta; q_b) + \lambda_{\text{ES}}\mathcal{L}_{\text{ES}}(\theta; q_b) + \lambda_{\text{VS}}\mathcal{L}_{\text{VS}}(\theta; q_b).
$$

Start with pure CFM ($\lambda_{\text{ES}} = \lambda_{\text{VS}} = 0$), then add low-weight proper scores on $M_{\text{grad}} = 6\text{--}8$ rollouts (comparing checkpointed full backprop vs. FlowBP-style sparse gradients in Exp M3).

## 8.1 Complete Explicit Formulas for Proper Scoring Rules

1. **Unbiased Energy Score:**

$$
\mathcal{L}_{\text{ES}} = \frac{1}{B}\sum_{t=1}^{B}\left[\frac{1}{M}\sum_{m=1}^{M}\frac{\|\hat{Y}_t^{(m)} - Y_t\|_2}{\sqrt{D}} - \frac{1}{2 M(M - 1)}\sum_{m \neq m'}\frac{\|\hat{Y}_t^{(m)} - \hat{Y}_t^{(m')}\|_2}{\sqrt{D}}\right].
$$

2. **Explicit Factored Structured Variogram Score ($p_{\text{var}} = 0.5$):**

For coordinate indices $a = (h_1, n_1)$ and $b = (h_2, n_2)$ in $\{1, \dots, H\} \times \{1, \dots, N\}$, define the squared pairwise variogram error on an $M$-member ensemble:

$$
d_{p_{\text{var}}}(a, b; Y, \hat{Y}) = \left(\left|Y_a - Y_b\right|^{p_{\text{var}}} - \frac{1}{M}\sum_{m=1}^{M}\left|\hat{Y}_a^{(m)} - \hat{Y}_b^{(m)}\right|^{p_{\text{var}}}\right)^2.
$$

Partition pairs into three explicit subsets:

* **Temporal subset $\mathcal{S}_{\text{temp}}$:** $\{((h_1, n), (h_2, n)) : 1 \le h_1 < h_2 \le H,\; 1 \le n \le N\}$, cardinality $|\mathcal{S}_{\text{temp}}| = N\binom{H}{2} = 9,500$.
* **Cross-asset subset $\mathcal{S}_{\text{asset}}$:** $\{((h, n_1), (h, n_2)) : 1 \le h \le H,\; 1 \le n_1 < n_2 \le N\}$, cardinality $|\mathcal{S}_{\text{asset}}| = H\binom{N}{2} = 24,500$.
* **Cross-time/cross-asset subset $\mathcal{S}_{\text{cross}}$:** Uniform random draw without replacement of $K_{\text{cross}} = 2,000$ distinct pairs $((h_1, n_1), (h_2, n_2))$ with $h_1 \ne h_2$ and $n_1 \ne n_2$ (using a **fixed deterministic pair tensor** for validation/test reproducibility, and freshly resampled uniformly per minibatch during training).

Normalize each component by its subset cardinality (default $\lambda_t = \lambda_a = \lambda_x = 1/3$):

$$
\mathcal{L}_{\text{VS}} = \frac{1}{B}\sum_{t=1}^{B}\left[\frac{\lambda_t}{|\mathcal{S}_{\text{temp}}|}\sum_{(a,b)\in\mathcal{S}_{\text{temp}}} d_{p_{\text{var}}}(a,b) + \frac{\lambda_a}{|\mathcal{S}_{\text{asset}}|}\sum_{(a,b)\in\mathcal{S}_{\text{asset}}} d_{p_{\text{var}}}(a,b) + \frac{\lambda_x}{|\mathcal{S}_{\text{cross}}|}\sum_{(a,b)\in\mathcal{S}_{\text{cross}}} d_{p_{\text{var}}}(a,b)\right].
$$

## 8.2 Diagnostics, Matched Checkpoints & Mandatory Post-Calibration Validation

1. **Three-point straightness oscillation diagnostic:** Log FP32 $\hat{C}_{\text{straight}}(P_{\text{new}}^{(b)}, v_{\text{old}}^{(b)})$ (after Block A), $\hat{C}_{\text{straight}}(P_{\text{new}}^{(b)}, v_{\text{new}}^{(b)})$ (after Block B), and $\hat{C}_{\text{straight}}(P_{\text{new}}^{(b+1)}, v_{\text{new}}^{(b)})$ (after next Block A).
2. **Dual validation & matched checkpoint pairs:** After every Block B, validate `ema_dit.eval()` using both $q_b$ and standalone $P_{\text{new}}^{(b)}$. Always save and load matched `(ema_prior, ema_dit)` pairs together.
3. **Final deployment calibration ($\rho = 1.0$) + post-calibration validation:** Load the best validation **pre-calibration pair**, freeze $P_{\text{deploy}} = P_{\bar{\psi}^*}$, run $5\text{--}10$ epochs of pure $P_{\text{deploy}}$-source CFM calibration, save the **post-calibration pair**, and validate it on the validation split. Select pre- vs. post-calibration strictly on validation before touching the chronological test split.
4. **Stage 6 optional solver-aware prior objective:** Evaluate $C_{\text{solver}}(\xi_0; h_t) = \frac{1}{D}\|G_{\text{Heun}, K_{\text{low}}}(\xi_0, h_t) - G_{\text{Heun}, K_{\text{ref}}}(\xi_0, h_t)\|_2^2$ ($K_{\text{low}} = 2, K_{\text{ref}} = 16$) strictly as a Stage 6 efficiency extension—always paired with independent probabilistic forecast scoring so solver stability is never confused with forecast accuracy.

---

# 9. Complete Reference Training Algorithm (PyTorch-Style Pseudocode)

```python
# ==============================================================================
# Modules:
#   market_encoder (E_phi):  Frozen after Stage 1 baseline CFM pretraining
#   online_prior (P_psi):    Dual-head bounded conditionally factorized Gaussian
#   ema_prior (P_psi_bar):   EMA copy of online_prior (alpha_P = 0.995)
#   online_dit (v_theta):    Trainable spatiotemporal DiT (p in {1, 2, 4})
#   ema_dit (v_theta_bar):   EMA copy of online_dit (alpha_G = 0.999)
# State shape: (B, H_p, N, p) with D = H_p * N * p = 1000
# ==============================================================================

def heun_rollout_fp32_with_straightness(vel_net, xi_0, ctx, num_steps=8):
    """Exact discrete Heun constant-velocity straightness; FP32 state/accum; O(D) mem wrt K."""
    z = xi_0.float()
    z0 = z
    dt = 1.0 / num_steps
    sum_v_sq = torch.zeros(z.shape[0], device=z.device, dtype=torch.float32)

    for m in range(num_steps):
        t0 = torch.full((z.shape[0],), m * dt, device=z.device, dtype=torch.float32)
        t1 = torch.full((z.shape[0],), (m + 1) * dt, device=z.device, dtype=torch.float32)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            k1 = vel_net(z, t0, ctx)
        k1_f = k1.float()
        z_pred = z + dt * k1_f

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            k2 = vel_net(z_pred, t1, ctx)
        k2_f = k2.float()
        z = z + 0.5 * dt * (k1_f + k2_f)

        sum_v_sq += k1_f.pow(2).mean(dim=(1, 2, 3))
        sum_v_sq += k2_f.pow(2).mean(dim=(1, 2, 3))

    mean_v_sq = sum_v_sq / (2.0 * num_steps)
    disp_sq = (z - z0).pow(2).mean(dim=(1, 2, 3))
    straight_cost = (mean_v_sq - disp_sq).clamp_min(0.0)
    return z, straight_cost


def compute_rloo_advantages(costs, prev_scale, antithetic=False, clip_val=None):
    """
    Score-function RLOO (unbiased prior to any advantage or gradient clipping).
    Uses strictly PREVIOUS-BATCH running std (prev_scale).
    """
    B, M = costs.shape
    denom = prev_scale + 1e-6

    if not antithetic:  # Canonical baseline: independent samples
        sum_c = costs.sum(dim=1, keepdim=True)
        baseline = (sum_c - costs) / (M - 1)
        adv = (costs - baseline) / denom
    else:               # Ablation M1: pair-level antithetic RLOO
        assert M % 2 == 0 and M >= 4
        Mp = M // 2
        c_pos, c_neg = costs[:, :Mp], costs[:, Mp:]
        c_pair_mean = 0.5 * (c_pos + c_neg)
        sum_pairs = c_pair_mean.sum(dim=1, keepdim=True)
        pair_baseline = (sum_pairs - c_pair_mean) / (Mp - 1)
        adv = torch.cat([(c_pos - pair_baseline) / denom,
                         (c_neg - pair_baseline) / denom], dim=1)

    if clip_val is not None:  # Biased-stabilization ablation only
        adv = adv.clamp(-clip_val, clip_val)
    return adv


beta_kl = 1.0
running_cost_std = RunningEMAStd(momentum=0.95, init_value=1.0)

for b in range(num_outer_steps):
    prior_ref = frozen_copy(ema_prior).eval()
    dit_ref = frozen_copy(ema_dit).eval()  # Deterministic rollouts

    # ==========================================================================
    # BLOCK A: GENERATOR-COUPLED PRIOR OPTIMIZATION
    # ==========================================================================
    set_trainable(online_prior, True)
    online_prior.train()
    set_trainable(online_dit, False)

    for epoch in range(prior_epochs):
        prior_loader.dataset.set_random_offset()  # Stride S in {5, 10, 20}
        epoch_kl_vals, clip_hits, total_steps = [], 0, 0

        for step_idx, (past_x, future_y_patch, meta) in enumerate(prior_loader):
            B = past_x.shape[0]
            with torch.no_grad():
                ctx = market_encoder(past_x)

            # 1. Strictly Proper Coordinate-Normalized Gaussian NLL
            nll_loss = -online_prior.log_prob_per_dim(future_y_patch, ctx).mean()

            # 2. Consistent Local Distributional Transport (ATTACHED rsample!)
            xi_dist = online_prior.rsample(ctx)
            if dist_loss_type == "sw2":
                dist_loss = local_mc_sliced_wasserstein2(
                    xi_dist, future_y_patch, ctx.global_emb, meta, num_proj=256
                )
            else:
                dist_loss = local_consistent_sinkhorn_divergence(
                    xi_dist, future_y_patch, ctx.global_emb, meta, eps_ot=eps_ot
                )

            # 3. Generator-Dependent Straightness PG (DETACHED samples, FP32 ODE!)
            with torch.no_grad():
                mu, sigma = online_prior.dist_params(ctx)
                if use_antithetic_pairs:
                    eps_half = torch.randn((B, M // 2, *mu.shape[1:]), device=mu.device)
                    eps = torch.cat([eps_half, -eps_half], dim=1)
                else:
                    eps = torch.randn((B, M, *mu.shape[1:]), device=mu.device)
                xi_pg = (mu.unsqueeze(1) + sigma.unsqueeze(1) * eps).detach()

                flat_xi = xi_pg.flatten(0, 1)
                flat_ctx = ctx.repeat_interleave(M, dim=0)
                _, flat_costs = heun_rollout_fp32_with_straightness(
                    dit_ref, flat_xi, flat_ctx, num_steps=heun_steps_block_a
                )
                costs = flat_costs.view(B, M)

                # Read previous-batch scale strictly BEFORE updating
                prev_scale = running_cost_std.value
                adv = compute_rloo_advantages(
                    costs, prev_scale,
                    antithetic=use_antithetic_pairs,
                    clip_val=adv_clip_ablation  # None in canonical unbiased mode
                )
                running_cost_std.update(costs)

            # Full joint summed log_prob (no double 1/D scaling)
            log_p_sum = online_prior.log_prob_sum(
                xi_pg.flatten(0, 1).detach(), flat_ctx
            ).view(B, M)
            straight_pg_loss = (adv.detach() * log_p_sum).mean()

            # 4. Moving-Reference Per-Coordinate KL
            kl_per_dim = conditional_kl_per_dim(online_prior, prior_ref, ctx).mean()
            epoch_kl_vals.append(kl_per_dim.detach().item())

            if epoch == 0 and step_idx == 0:
                log_grad_norm_ratio(online_prior, lambda_straight * straight_pg_loss,
                                    lambda_nll * nll_loss)

            prior_loss = (
                lambda_nll * nll_loss
                + lambda_dist * dist_loss
                + lambda_straight * straight_pg_loss
                + beta_kl * kl_per_dim
            )

            prior_opt.zero_grad(set_to_none=True)
            prior_loss.backward()
            grad_norm = clip_grad_norm_(online_prior.parameters(), max_norm=1.0)
            clip_hits += int(grad_norm > 1.0)
            total_steps += 1
            prior_opt.step()
            update_ema(ema_prior, online_prior, decay=alpha_P)

        mean_epoch_kl = sum(epoch_kl_vals) / len(epoch_kl_vals)
        beta_kl = float(
            np.clip(beta_kl * math.exp(eta_beta * (mean_epoch_kl - delta_kl)), 0.05, 50.0)
        )
        log_prior_epoch(b, epoch, mean_epoch_kl, beta_kl, clip_rate=clip_hits / total_steps)

    prior_new = frozen_copy(ema_prior).eval()
    source_mixture = FrozenMixture([prior_new, prior_ref], weights=[rho, 1.0 - rho])
    c_straight_pre_b = eval_mean_straightness_fp32(prior_new, dit_ref, market_encoder, val_loader)

    # ==========================================================================
    # BLOCK B: SOURCE-MATCHED GENERATOR OPTIMIZATION
    # ==========================================================================
    set_trainable(online_prior, False)
    set_trainable(online_dit, True)
    online_dit.train()

    for epoch in range(generator_epochs):
        for past_x, future_y_patch, meta in generator_loader:
            with torch.no_grad():
                ctx = market_encoder(past_x)
                xi_0 = source_mixture.sample(ctx).detach()

            # Same-context baseline OR Symmetric-KL-Masked Log-Domain Sinkhorn OT
            # Uniform marginals a=b=1/B_g; fallback to diagonal on convergence failure; paired_ctx=ctx[j]
            xi_pair, y_pair, ctx_pair, ot_diag = make_cfm_pairs(
                xi_0, future_y_patch, ctx, source_mixture, meta,
                mode=coupling_mode, delta_context=delta_context
            )

            tau = sample_flow_time(xi_pair.shape[0], mode=time_sampling_mode)
            tau_view = tau.view(-1, 1, 1, 1)
            z_tau = (1.0 - tau_view) * xi_pair + tau_view * y_pair
            u_target = y_pair - xi_pair

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred = online_dit(z_tau, tau, ctx_pair)
                cfm_loss = (v_pred.float() - u_target.float()).pow(2).mean()

            if lambda_es > 0.0 or lambda_vs > 0.0:
                ens_patch = differentiable_ode_rollout_fp32(
                    online_dit, source_mixture, ctx, num_samples=M_grad,
                    steps=heun_steps_block_b, mode=ode_backprop_mode
                )
                ens_y, true_y = unpatchify(ens_patch), unpatchify(future_y_patch)
                es_loss = unbiased_energy_score(ens_y, true_y)
                vs_loss = factored_variogram_score(ens_y, true_y, fixed_cross_pairs=None)
            else:
                es_loss = vs_loss = 0.0

            gen_loss = lambda_cfm * cfm_loss + lambda_es * es_loss + lambda_vs * vs_loss

            gen_opt.zero_grad(set_to_none=True)
            gen_loss.backward()
            clip_grad_norm_(online_dit.parameters(), max_norm=1.0)
            gen_opt.step()
            update_ema(ema_dit, online_dit, decay=alpha_G)

    ema_dit_eval = frozen_copy(ema_dit).eval()
    c_straight_post_b = eval_mean_straightness_fp32(prior_new, ema_dit_eval, market_encoder, val_loader)

    # Dual Validation + Saturation + Multi-K Rank Stability + MC Symmetric Mixture KL (S=128)
    val_mix = validate_ensemble(source_mixture, ema_dit_eval, market_encoder, val_loader)
    val_new = validate_ensemble(prior_new, ema_dit_eval, market_encoder, val_loader)
    sat_stats = log_sigma_saturation(prior_new, market_encoder, val_loader)
    rank_corr = eval_multi_k_straightness_rank_corr(prior_new, ema_dit_eval, market_encoder, val_loader)
    mc_mix_kl, mc_mix_se = eval_mc_sym_mixture_kl(source_mixture, market_encoder, val_loader, num_mc=128)
    log_diagnostics(b, beta_kl, c_straight_pre_b, c_straight_post_b, val_mix, val_new,
                    sat_stats, rank_corr, ot_diag, mc_mix_kl, mc_mix_se)
    checkpoint_matched_pair_if_improved(val_new, ema_prior, ema_dit, tag="pre_calibration")

# ==============================================================================
# FINAL DEPLOYMENT CALIBRATION (rho = 1.0) + MANDATORY POST-CALIBRATION VALIDATION
# ==============================================================================
load_matched_checkpoint_pair(ema_prior, ema_dit, tag="pre_calibration")
prior_deploy = frozen_copy(ema_prior).eval()
online_dit.load_state_dict(ema_dit.state_dict())

run_final_cfm_calibration(online_dit, ema_dit, prior_deploy, market_encoder,
                          generator_loader, epochs=calibration_epochs)

val_post_calib = validate_ensemble(prior_deploy, ema_dit.eval(), market_encoder, val_loader)
save_matched_checkpoint_pair(prior_deploy, ema_dit, val_post_calib, tag="post_calibration")
# Run untouched chronological test evaluation ONLY after selecting pre- vs post-calib on val!
```

---

# 10. Inference, Evaluation Triad, Ablation Matrix & Dual Compute Parity

## 10.1 Inference Pipeline

At test time, zero target information or reward selection is accessed:

1. Encode market history: $h_t = E_\phi(X_t^{\text{past}})$.
2. Draw $M$ initial source latents from the validated deployment prior: $\xi_0^{(i)} \sim P_{\text{deploy}}(\cdot \mid h_t)$.
3. Integrate the calibrated EMA DiT $v_{\bar{\theta}}$ (in `.eval()` mode, FP32 ODE state) from $\tau = 0$ to $\tau = 1$ at the target NFE budget to obtain $z^{(i)}(1)$.
4. Unpatchify $\hat{Y}_t^{(i)} = \mathcal{P}_p^{-1}\!\big(z^{(i)}(1)\big) \in \mathbb{R}^{H \times N \times 1}$.
5. Reconstruct future price trajectories using strictly causal scale $s_{n,t}$:

$$
\hat{P}_{n, t + h}^{(i)} = P_{n, t} \exp\!\left(s_{n, t}\sum_{j=1}^{h}\hat{\tilde{r}}_{n, t + j}^{(i)}\right).
$$

## 10.2 Three-Axis Evaluation Triad & Standardized NFE Grid

Evaluate all models on untouched chronological test periods using block-bootstrap / Diebold–Mariano tests accounting for overlapping $20$-day horizons:

1. **Transport Efficiency:** Coupling-indexed displacement $D_{\text{transport}}(\pi)$, local Sinkhorn Divergence / $\widehat{\text{SW}}_2^2$ (across multiple projection seeds), FP32 constant-velocity straightness $\hat{C}_{\text{straight}}^{(K)}$ across $K \in \{4, 8, 16, 32\}$, pre/post-Block B straightness stability, $\sigma_\psi$ boundary saturation, and exploratory correlation between marginal $\sigma_\psi(h_t)$ and scale-normalized realized future volatility $\sigma_{i,t}^{\text{future, norm}} = \sigma(r_{i, t+1:t+H}) / s_{i,t}$.
2. **Computational Efficiency (Primary Pareto Frontier):** Evaluate fixed-step solvers at compatible function-evaluation budgets:

$$
\text{NFE} \in \{4, 8, 16, 32, 64\}
$$

mapping to Euler ($1$ NFE/step $\Rightarrow K \in \{4, 8, 16, 32, 64\}$), Heun ($2$ NFE/step $\Rightarrow K \in \{2, 4, 8, 16, 32\}$), and RK4 ($4$ NFE/step $\Rightarrow K \in \{1, 2, 4, 8, 16\}$), plus realized NFE and wall-clock latency for adaptive Dormand–Prince. Also log endpoint discrepancy against a $64$-NFE reference. Primary figure: **Probabilistic Score vs. NFE and Wall-Clock Latency**.
3. **Probabilistic Forecasting & Financial Realism:** Unbiased Energy Score, Explicit Factored Variogram Score (fixed validation/test $\mathcal{S}_{\text{cross}}$), marginal CRPS, $90\%$ and $50\%$ interval coverage & width, PIT histograms, ensemble mean/median RMSE & MAE (best-of-$M$ strictly excluded), generated endpoint volatility calibration, ACF of squared returns, tail quantiles/kurtosis, max drawdowns, and Ledoit–Wolf / sector correlations across calm vs. high-volatility test sub-periods.

## 10.3 Complete Ablation Matrix (Including Experiment R & Dual Compute Parity)

**Dual Compute-Parity Rule for Exp I vs. Exp G:** Report two separate comparisons between Exp I (Two-Stage Sequential) and Exp G (Alternating):
* **Comparison A (Optimizer-step matched):** Identical prior and DiT gradient step counts.
* **Comparison B (Compute matched):** Equal total GPU wall-clock training time and estimated FLOPs (accounting for Block A's $2KM$ frozen DiT rollouts), plotted as **Probabilistic Score vs. Total Computational Budget**.

| ID | Configuration | Isolates / Tests |
|---|---|---|
| **A** | Standard $\mathcal{N}(0, I)$ source + standard MSE CFM | Primary generative baseline |
| **R** | **Direct Prior-Only Forecasting ($0$ NFE):** sample $\hat{Y} = \mathcal{P}_p^{-1}(\xi_0)$ from $P_\psi$ (NLL-only and NLL+$\widehat{\text{SW}}_2^2$) without DiT | Quantifies exact joint multivariate value added by the Flow Matching DiT |
| **B** | Two-Stage Sequential: Gaussian NLL prior $\rightarrow$ freeze $\rightarrow$ CFM | Value of conditional marginal NLL prior alone |
| **C** | Two-Stage Sequential: Local $\widehat{\text{SW}}_2^2$ / Sinkhorn prior $\rightarrow$ freeze $\rightarrow$ CFM | Value of local distributional transport alone |
| **D** | Alternating with **only** Straightness PG + KL (no NLL/dist) | Whether straightness alone collapses or stabilizes |
| **E** | Simultaneous prior + generator updates | Coupled non-stationarity vs. alternation |
| **F** | Alternating with fixed $\mathcal{N}(0, I)$ KL reference | Moving KL reference vs. permanent Gaussian anchor |
| **G** | **Full v4.3 Proposed Method** (Alternating + Straightness PG + Moving KL + Dual EMA) | Complete proposed system |
| **H** | Full v4.3 with $\lambda_{\text{straight}} = 0$ (no generator coupling in Block A) | Proves generator coupling drives alternating gains |
| **I** | **Two-Stage Sequential Baseline** (Comparison A: step-matched; Comparison B: FLOPs/wall-clock-matched) | Mandatory falsification test for alternating optimization |
| **J** | Full v4.3 without EMA networks | Value of dual EMA snapshots |
| **K** | Full v4.3 without KL trust region ($\beta_b = 0$) | Value of trust-region control |
| **L** | **Primary Coupling Study:** Same-context independent coupling vs. Symmetric-KL-Masked Local OT Approximation | Core transport-efficiency coupling study |
| **M1** | Independent RLOO vs. Pair-Level Antithetic RLOO vs. Token-Level RLOO across $M \in \{20, 64, 128\}$ (plus clipping ablation) | Policy-gradient variance, bias, and rollout-cost trade-off |
| **M2** | Uniform vs. Logit-Normal flow-time sampling | Flow-time allocation ablation |
| **M3** | Full checkpointed ODE backprop vs. truncated / FlowBP-style sparse gradients (when $\lambda_{\text{ES}}, \lambda_{\text{VS}} > 0$) | Auxiliary score gradient compute efficiency |
| **N** | **Source-Mismatch Negative Control:** Adapted prior at rollout, $\mathcal{N}(0, I)$ in CFM training | Directly verifies necessity of source-matched CFM |
| **O** | Patch size $p \in \{1, 2, 4\}$ and Block A stride $S \in \{5, 10, 20\}$ | Temporal granularity and target-overlap sensitivity |
| **P** | MSE CFM vs. Pseudo-Huber CFM vs. Tail-Weighted MSE | Robustness vs. exact conditional-mean velocity field |
| **Q** | Constant-velocity straightness vs. Solver-aware consistency $\mathcal{L}_{\text{solver}}$ & Low-Rank Factor Prior $\Sigma_\psi = D_\psi + B_\psi B_\psi^\top$ | Stage 6 efficiency & prior expressivity extensions |

## 10.4 Separate Mode: Reward-Tilted Scenario Generation

Unbiased probabilistic forecasting must never be conflated with stress-testing. To generate targeted scenarios (crashes, volatility spikes, correlation breakdowns), explicitly tilt the training target distribution $p_R(Y^{\text{patch}} \mid h_t) \propto p_{\text{data}}(Y^{\text{patch}} \mid h_t)\exp(R(Y, h_t)/T_R)$ and train a separate importance-weighted CFM model toward $p_R$, evaluating it strictly as scenario generation rather than calibrated forecasting.

---

# 11. Concrete Engineering Implementation Plan

## 11.1 Repository & Modular Codebase Layout

Design the codebase so Stage 1 runs completely standalone while leaving clean, zero-refactor extension hooks for Stages 2--6:

```text
finflow_dit/
├── configs/
│   ├── base_data.yaml             # Universe, cal_prefix=252, L=60, H=20, purge=25, p in {1,2,4}
│   ├── stage1_baseline_cfm.yaml   # Exp A: N(0,I) source, MarketEncoder + PatchedFlowDiT
│   ├── stage2_static_prior.yaml   # Exps R, B, C: Dual-head prior + NLL + SW2/Sinkhorn
│   ├── stage3_coupling_ot.yaml    # Exp L: Symmetric-KL masked log-domain Sinkhorn OT
│   ├── stage4_straightness.yaml   # Exp M1: Frozen EMA DiT + FP32 Heun straightness RLOO
│   └── stage5_alternating.yaml    # Exps G, H, I: Full alternating loop + dual parity + rho=1 calib
├── data/
│   ├── download_and_clean.py      # Adjusted daily OHLCV + index/sector alignment + corporate-action checks
│   ├── preprocessing.py           # Cal-prefix s_min, past-only s_{i,t}, 12 features, LW correlation graphs A_t
│   ├── dataset.py                 # Chronological splits, 25-day purge, Patcher(p), Block A strided random-offset loader
│   └── test_leakage.py            # Pytest suite: future-perturbation 0.0 delta test + shape/purge assertions
├── models/
│   ├── market_encoder.py          # Temporal Transformer + Signed-Edge GATv2 + AttentionPool -> (h_stocks, h_global)
│   ├── patched_dit.py             # Configurable 250/500/1000-token spatiotemporal DiT + AdaLN-Zero + zero-init head
│   ├── source_prior.py            # Dual-head bounded Gaussian (mu_head bias=0, scale_head bias=b0), log_prob_sum/per_dim
│   └── ema_utils.py               # EMA wrappers, frozen_copy, FrozenMixture, matched (prior, dit) pair checkpointing
├── flow/
│   ├── ode_solvers.py             # FP32 Euler/Heun/RK4/Dopri5 + exact Heun straightness + NFE/wall-clock profiler
│   ├── transport.py               # Local neighborhoods, consistent Sinkhorn Div, MC SW2^2, symmetric-KL masked OT + fallback
│   ├── rloo.py                    # Independent & Pair-Level Antithetic RLOO + past-batch RunningEMAStd + grad-norm logger
│   └── metrics.py                 # Unbiased Energy Score, Explicit 3-subset Variogram Score, CRPS, PIT, coverage, LW corr
├── baselines/
│   └── classical.py               # Historical block bootstrap + Multivariate EWMA/DCC-GARCH benchmark
├── scripts/
│   ├── train_stage1_cfm.py        # Stage 1: Baseline Gaussian CFM + NFE={4,8,16,32,64} sweep
│   ├── train_stage2_prior.py      # Stage 2: Static prior + Exp R (0-NFE) + Exps B/C
│   ├── train_stage3_coupling.py   # Stage 3: Independent vs. KL-masked local OT study
│   ├── train_stage4_rloo_check.py # Stage 4: Frozen-generator straightness PG verification
│   ├── train_stage5_alt.py        # Stage 5: Full alternating loop + post-rho=1 calibration + Exp I parity
│   └── evaluate_test.py           # Untouched test-set evaluation, Diebold-Mariano/block-bootstrap, calm vs. crash audit
└── tests/
    ├── test_math_identities.py    # Heun straightness == 0 on constant vel; dual-head init == N(0,I); RLOO zero-mean grad check
    └── test_ot_and_metrics.py     # Diagonal Sinkhorn feasibility + fallback; SW2^2 == 0 for P=Q; Variogram exactness check
```

## 11.2 Fast-Prototyping & Compute Protocol

1. **Tech stack:** PyTorch 2.x (`torch.compile` on DiT blocks), `bf16` autocast for forward/backward DiT passes with strictly `fp32` ODE states/accumulators/Sinkhorn dual potentials, Hydra configs, Weights & Biases (`wandb`) logging, and `pytest` for mathematical unit tests.
2. **Small-scale smoke track first:** Before running $N = 50$ full grids, run Stages 1--5 on a fast **$N = 15$ liquid-equity subset** ($D = 300$, $75$ DiT tokens at $p = 4$) on a single 24GB GPU (e.g., RTX 3090/4090 or A10G). Every mathematical identity, gradient flow, and Stage 1--5 gate should pass end-to-end in hours before scaling to $N = 50$.
3. **Compute accounting:** Log exact wall-clock seconds, peak VRAM, total DiT forward/backward FLOPs, and total NFE for every run so Comparison B (Compute-Matched Exp I vs. Exp G) is automatic rather than reconstructed post-hoc.

## 11.3 Gated 6-Stage Execution Schedule, Deliverables & Mandatory Exit Gates

Do **not** advance to the next stage until every exit gate of the current stage passes on the validation split.

| Stage | Modules Built & Executed | Mandatory Unit Tests & Diagnostics | Strict Quantitative Exit Gate |
|---|---|---|---|
| **Stage 1: Data, Classical Baselines & Baseline Patched CFM (Exp A)** | `preprocessing.py`, `dataset.py`, `test_leakage.py`, `classical.py`, `market_encoder.py`, `patched_dit.py`, `ode_solvers.py`, `metrics.py`, `train_stage1_cfm.py` | Zero-leakage perturbation test (`0.0` delta on $X_t, s_{i,t}, A_t$); Heun straightness $= 0$ on synthetic constant-velocity field; GNN signed edge sparsity log. | All leakage & math tests pass; Exp A beats Historical Bootstrap on validation Energy Score & Factored Variogram Score; $90\%$ interval coverage inside $[85\%, 95\%]$; baseline Pareto curve logged over $\text{NFE} \in \{4, 8, 16, 32, 64\}$. |
| **Stage 2: Static Conditional Prior & Direct Prior Baseline (Exps R, B, C)** | `source_prior.py`, `transport.py` (local neighborhoods, consistent Sinkhorn Div, MC $\widehat{\text{SW}}_2^2$), `train_stage2_prior.py` | Dual-head init asserts $\mu_\psi \equiv 0, \sigma_\psi \equiv 1$; $\widehat{\text{SW}}_2^2$ convergence across $L_{\text{proj}} \in \{128, 256, 512\}$ and multiple seeds; prior boundary saturation check. | Confirmed reduction in source-to-target displacement vs. $\mathcal{N}(0, I)$; $\sigma_\psi$ saturation $< 5\%$; Exp A / B / C clearly beat Exp R ($0$-NFE prior-only) on Factored Variogram Score & LW correlation error (proving DiT value); generated endpoint volatility calibrated. |
| **Stage 3: Primary Coupling Study (Exp L)** | `transport.py` (symmetric convexity upper-bound mask, masked log-domain Sinkhorn, uniform marginals, fallback), `train_stage3_coupling.py` | Diagonal feasibility & fallback unit test; log off-diagonal match rate, Sinkhorn marginal error $< 10^{-4}$, fallback rate, and $S = 128$ MC symmetric mixture KL ($\pm$ SE). | Clean comparison of coupling-indexed $D_{\text{transport}}(\pi)$, straightness cost, and validation proper scores for Independent vs. KL-Constrained Local OT; choose canonical coupling for Stages 4--5. |
| **Stage 4: Generator-Dependent Straightness PG Verification (Exp M1)** | `rloo.py` (Independent RLOO, Pair-Level Antithetic RLOO, past-batch `RunningEMAStd`, clip-rate & grad-norm ratio logger), `train_stage4_rloo_check.py` | Synthetic zero-mean score-function unbiasedness test (before clipping); Spearman rank correlation of $\hat{C}_{\text{straight}}^{(K)}$ across $K \in \{4, 8, 16, 32\}$; `dit_ref.eval()` determinism check. | Monotonic drop in validation $\hat{C}_{\text{straight}}^{(K)}$ under frozen EMA DiT; multi-$K$ Spearman rank correlation $> 0.85$; $\lambda_{\text{straight}}$ calibrated via live grad-norm ratio; clipping activation logged ($< 10\%$). |
| **Stage 5: Full Alternating Loop vs. Compute-Matched Sequential (Exps G, H, I)** | `ema_utils.py` (dual EMAs, `FrozenMixture`, matched pair checkpoints), `train_stage5_alt.py` (Block A/B loop, epoch dual $\beta_b$, dual validation, $\rho = 1$ final calibration) | Three-point straightness oscillation log (pre-B $\to$ post-B $\to$ next-A); pre- vs. post-calibration validation check; dual parity plots (step-matched AND wall-clock/FLOPs-matched Exp I vs. G). | Stable per-dim KL tracking $\delta$; Exp G achieves lower straightness cost and matches or improves low-NFE ($4\text{--}8$ NFE) Energy & Variogram Scores vs. Exp H ($\lambda_{\text{straight}} = 0$) and compute-matched Exp I without losing calibration. |
| **Stage 6: Efficiency Extensions, Full Ablations & Untouched Test Audit** | `evaluate_test.py` + Exps D, E, F, J, K, M2, M3 (FlowBP), N (source-mismatch control), O ($p \in \{1,2,4\}$, $S \in \{5,10,20\}$), P (Huber), Q ($\mathcal{L}_{\text{solver}}$, low-rank prior) | Multi-seed runs ($\ge 3$ seeds); Diebold–Mariano & block-bootstrap significance tests; calm vs. high-volatility/crash test regime slices. | Final untouched test-set Pareto frontier (Probabilistic Score vs. NFE & Wall-Clock Time) and full falsifiable ablation table completed. |

## 11.4 Initial Default Hyperparameter Table

| Parameter | Default Value | Parameter | Default Value |
|---|---|---|---|
| Universe / Cal Prefix / Purge | $N = 50$ ($N = 15$ smoke), $T_{\text{cal}} = 252$, $25$ days | Lookback / Horizon / Features | $L = 60$, $H = 20$, $F = 12$ |
| Patch size $p$ / State dim $D$ | $p = 4$ (ablate $1, 2$), $D = 1,000$ | DiT width / Layers / Heads | $d_{\text{model}} = 128$, $L_{\text{DiT}} = 6$, $8$ heads |
| Prior bounds $(\sigma_{\min}, \sigma_{\max})$ | $(0.20, 4.00)$, dual-head init $\mathcal{N}(0, I)$ | Block A rollouts $M$ / Stride $S$ | $M = 20$ (ablate $64, 128$), $S = 5$ (ablate $10, 20$) |
| Eval NFE grid / Block A Heun $K$ | $\text{NFE} \in \{4, 8, 16, 32, 64\}$ / $K = 8$ | Local group $B_g$ / $\widehat{\text{SW}}_2^2$ projs | $B_g = 24$, $L_{\text{proj}} = 256$ (ablate $128, 512$) |
| KL budget $\delta$ / Dual step $\eta_\beta$ | $0.03$ nats/coord, $\eta_\beta = 0.5$ | Context OT KL threshold $\delta_{\text{context}}$ | $0.05$ nats/coord (symmetric upper bound) |
| Mixture weight $\rho$ | $0.70$ (final calib $\rho = 1.0$) | Loss weights $(\lambda_{\text{NLL}}, \lambda_{\text{dist}}, \lambda_{\text{straight}})$ | $(1.0, 0.5, \text{calibrated via grad-norm ratio})$ |
| EMA decays $(\alpha_P, \alpha_G)$ | $(0.995, 0.999)$ | Outer iterations / Epochs per block | $B_{\text{outer}} = 6\text{--}10$, $5\text{--}8$ epochs |
