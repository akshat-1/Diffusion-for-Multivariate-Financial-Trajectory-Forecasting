# Reward-Guided Latent Diffusion for Multivariate Financial Trajectory Forecasting

## Complete architecture and mathematical training specification

**Proposed training strategy:** Alternating prior–generator optimization with a moving learned-prior reference, KL trust-region regularization, and separate exponential moving average (EMA) networks.

---

## Abstract

This document specifies a generative forecasting architecture for multivariate financial time series. Given a historical window of data for multiple stocks, the system learns a representation of the current market, generates a distribution of possible future trajectories using a conditional latent Diffusion Transformer (DiT), and progressively adapts the diffusion model's initial-noise distribution using trajectory-quality feedback.

The architecture contains five principal learned components:

1. A **historical market encoder** that represents temporal patterns, cross-stock relationships, and broader market conditions.
2. A **future trajectory autoencoder** that maps observed future trajectories into a compact latent space and reconstructs them from that space.
3. A **conditional latent DiT** that learns to generate future latent trajectories from a noisy terminal latent and historical context.
4. A **conditional terminal prior** that learns which initial diffusion-latent regions are useful for a given historical market context.
5. A **reward evaluator** that scores generated trajectories using ground-truth forecasting error and financial distributional properties during training.

The key optimization proposal is not to update the prior and generator simultaneously. Training alternates between two blocks:

- **Prior block:** freeze a snapshot of the generator and update the conditional terminal prior using reward-ranked samples. Regularize the new prior against the previous iteration's learned prior, not permanently against the original standard Gaussian.
- **Generator block:** freeze the newly adapted and previous prior snapshots, sample from a fixed mixture of these priors, and update the DiT and trajectory decoder using diffusion, reconstruction, and reward-based objectives.

Separate EMA copies provide stable sampling and reference networks. A moving-reference KL trust region limits how far the prior can move in one outer iteration, while the generator receives a dedicated optimization block to adapt to the updated sampling distribution.

The central research question is:

> Can progressive adaptation of a context-dependent diffusion terminal prior, coupled with alternating reward-guided optimization of a conditional latent diffusion generator, improve multivariate financial trajectory forecasting while preserving forecast calibration, cross-asset dependencies, and ensemble diversity?

This is a proposed research design, not a claim that the method is guaranteed to outperform standard diffusion. Its value must be established through controlled ablations and chronological out-of-sample evaluation.

---

# 1. Problem definition and notation

## 1.1 Forecasting task

Suppose the dataset contains historical observations for `N` stocks. At prediction date `t`, the model observes the previous `L` trading days and generates possible trajectories for the next `H` trading days.

The historical input is:

\[
X_t^{\mathrm{past}} \in \mathbb{R}^{L\times N\times F},
\]

where:

- `L` is the lookback length;
- `N` is the number of assets;
- `F` is the number of historical features per asset.

The future target is:

\[
Y_t^{\mathrm{future}} \in \mathbb{R}^{H\times N\times F_y}.
\]

For the first implementation, use `F_y = 1` and predict normalized daily log returns. Additional targets, such as realized volatility or volume changes, can be added later.

The system does not produce only one deterministic forecast. For a given historical context it generates an ensemble:

\[
\hat Y_t^{(1)},\hat Y_t^{(2)},\ldots,\hat Y_t^{(M)},
\]

where each member is a possible future trajectory for all modeled stocks.

The goal is to model the conditional distribution of future market trajectories, not merely to minimize the error of one selected path.

## 1.2 Notation table

| Symbol | Meaning |
|---|---|
| `X_t` | Historical input window ending at date `t` |
| `Y_t` | Ground-truth future trajectory after date `t` |
| `h_t` | Encoded historical market context |
| `z_0` | Clean latent encoding of a future trajectory |
| `z_k` | Noisy latent at forward/reverse diffusion step `k` |
| `z_K` | Terminal latent used to initialize reverse diffusion |
| `η` | Gaussian noise added by the forward diffusion process |
| `E_φ` | Historical market encoder with parameters `φ` |
| `Q_χ` | Future trajectory encoder / VAE posterior with parameters `χ` |
| `D_ω` | Future trajectory decoder with parameters `ω` |
| `R_θ` | Reverse diffusion process implemented by the DiT, parameters `θ_D` |
| `G_θ` | Complete latent generator: reverse DiT plus trajectory decoder |
| `P_ψ` | Learnable conditional terminal prior with parameters `ψ` |
| `\bar ψ` | EMA parameters of the conditional prior |
| `\bar θ` | EMA parameters of the complete generator |
| `b` | Outer prior–generator optimization iteration |
| `M` | Number of candidate trajectories generated per context |
| `K_e` | Number of elite trajectories selected for a sample-level update |

## 1.3 Two distinct diffusion-noise variables

It is essential to distinguish two kinds of noise.

**Forward-process noise** is denoted by `η`:

\[
\eta\sim\mathcal N(0,I).
\]

It is added to a clean future latent during diffusion training. The DiT is trained to predict this noise (or an equivalent target such as velocity).

**Terminal reverse-process latent** is denoted by `z_K`:

\[
z_K\sim P_\psi(z_K\mid h_t).
\]

It initializes the reverse diffusion process during generation. In a conventional diffusion model this is normally a standard Gaussian. The proposed research changes this terminal sampling distribution to a learnable context-dependent prior.

These two variables are not interchangeable. The reward-guided prior adaptation acts on `z_K`, not on the forward-process noise `η` used to construct training pairs.

There is also a separate VAE prior on the clean latent `z_0`. It is not automatically changed by adapting the diffusion terminal prior.

---

# 2. Dataset construction and financial preprocessing

## 2.1 Initial dataset scope

A practical first experiment can use approximately 30–50 liquid equities from one market, such as Indian equities. The universe can be expanded after the full pipeline works.

Possible data sources include:

- [NSE historical reports](https://www.nseindia.com/all-reports)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Alpha Vantage](https://www.alphavantage.co/)
- [FRED](https://fred.stlouisfed.org/) for macroeconomic series

Daily data is sufficient for the initial research question. Intraday order-book data is not required.

## 2.2 Candidate input features

| Feature group | Example features | Purpose |
|---|---|---|
| Price | Open, high, low, close | Price movement and daily range |
| Returns | Daily and multi-day log returns | Return dynamics |
| Volume | Volume, log-volume change, turnover | Trading activity |
| Volatility | Rolling volatility, downside volatility | Market uncertainty |
| Momentum | 5-, 20-, and 60-day momentum | Short/medium-term patterns |
| Market | Broad index returns and volatility | Market-wide context |
| Sector | Sector-index returns | Sector-level behaviour |
| Fundamentals | Point-in-time valuation and earnings features | Company characteristics |
| Macro | Rates, inflation, currency, commodities | Economic context |

Start with a compact feature set: returns, volume changes, high–low range, rolling volatility, momentum, broad-market returns, and sector returns. Fundamentals and macroeconomic variables can be added after the base model is established.

## 2.3 Return transformation and scaling

Raw stock prices have very different scales. Use adjusted prices and calculate log returns:

\[
r_{i,t}=\log\left(\frac{P_{i,t}}{P_{i,t-1}}\right),
\]

where `P_{i,t}` is the appropriately adjusted price for asset `i` at time `t`.

Estimate a rolling volatility scale using only information available by time `t`:

\[
s_{i,t}=\sqrt{\frac{1}{W-1}
\sum_{u=t-W+1}^{t}(r_{i,u}-\bar r_{i,t})^2+\varepsilon}.
\]

A normalized historical return is:

\[
\tilde r_{i,t}=\frac{r_{i,t}}{s_{i,t}}.
\]

For a future target after prediction date `t`, use the scale estimated at `t`:

\[
\tilde r_{i,t+h}=\frac{r_{i,t+h}}{s_{i,t}},\qquad h=1,\ldots,H.
\]

Do not normalize a future target using statistics calculated from that future period. The future target must be scaled only using information known at the prediction date.

The original scale `s_{i,t}` can also be supplied as an input feature. At inference, generated normalized returns can be converted back to log returns using the known historical scale.

## 2.4 Rolling-window examples

For each valid prediction date `t`, construct:

\[
X_t=[X_{t-L+1},\ldots,X_t]
\]

and:

\[
Y_t=[Y_{t+1},\ldots,Y_{t+H}].
\]

The dataset is:

\[
\mathcal D=\{(X_t,Y_t)\}_{t\in\mathcal T}.
\]

Example initial dimensions:

- Lookback `L = 60` trading days.
- Horizon `H = 20` trading days.
- Assets `N = 50`.
- Historical features `F = 10–15`.
- Future output features `F_y = 1` (normalized return).

Then:

\[
X_t\in\mathbb R^{60\times50\times F},
\qquad
Y_t\in\mathbb R^{20\times50\times1}.
\]

## 2.5 Leakage and survivorship controls

The dataset must be split chronologically into training, validation, and test periods. Use a purge or temporal gap around split boundaries when target windows overlap.

Required controls:

1. Use adjusted prices and correctly handle splits, dividends, and other corporate actions.
2. Use point-in-time index constituents where possible; do not assume today's index membership applied historically.
3. Use fundamentals only after their actual publication dates.
4. Fit normalization parameters on training data or using past-only rolling estimates.
5. Ensure the encoder at date `t` cannot access observations after `t`.
6. Ensure reward calculations use the subsequent target only during training or validation, never as an inference input.

---

# 3. Historical financial representation encoder

The historical encoder should represent three complementary aspects of the market:

1. Individual stock temporal dynamics.
2. Relationships between stocks.
3. Global market conditions.

The recommended initial encoder combines a temporal Transformer, a graph encoder, and a market-context encoder.

## 3.1 Temporal encoder

For stock `i`, extract its historical sequence:

\[
X_{t,i}\in\mathbb R^{L\times F}.
\]

A temporal Transformer processes it:

\[
u_{t,i}=T_\phi(X_{t,i}).
\]

The temporal encoder can capture momentum, volatility changes, temporal dependencies, and recurring patterns. Positional embeddings should identify the order of historical observations.

The output can be one vector per stock or a sequence of temporal tokens. A pooled vector per stock is sufficient for an initial graph model; retaining temporal tokens gives the later DiT richer conditioning information.

## 3.2 Cross-stock graph encoder

Represent the market as a time-dependent graph:

\[
G_t=(V,E_t),
\]

where each node represents a stock and edges represent statistical or learned relationships.

Let:

\[
A_t\in\mathbb R^{N\times N}
\]

be the adjacency or edge-weight matrix. A simple baseline is a rolling correlation graph constructed entirely from the historical input window. Later alternatives include partial-correlation graphs, transfer entropy, or learned edge weights.

Apply a graph neural network:

\[
v_{t,i}=\operatorname{GNN}_\phi(u_{t,i},A_t).
\]

This allows each stock representation to incorporate information from related stocks.

**Interpretation caution:** Correlation and predictive lead–lag relationships do not, by themselves, establish economic causation.

## 3.3 Global market-context encoder

Pool the stock representations to form a market-level summary:

\[
g_t=\operatorname{AttentionPool}(v_{t,1},\ldots,v_{t,N}).
\]

Then calculate a contextual market embedding:

\[
m_t=M_\phi(g_t).
\]

This representation can capture broad market volatility, market breadth, cross-sector dependence, and latent market regimes without requiring manually assigned bull/bear labels.

## 3.4 Fusion and final context

The final historical representation is:

\[
\boxed{h_t=E_\phi(X_t)=\operatorname{Fusion}
(u_{t,1:N},v_{t,1:N},m_t).}
\]

It is useful to preserve both stock-level tokens and a global token. For example:

\[
h_t\in\mathbb R^{(N+1)\times d},
\]

where `d` is the embedding dimension. The global token can condition the prior, while stock-level tokens can be supplied to the DiT through cross-attention.

## 3.5 Training the historical encoder

Initially, the encoder can be trained jointly with the baseline autoencoder/diffusion forecasting system. Optional self-supervised objectives include:

- Masked financial-feature reconstruction.
- Contrastive learning across two views of the same historical window.
- Prediction of historical summary statistics.

For the first reward-guided prior–generator experiments, freeze the historical encoder. This avoids introducing a third moving component while studying prior–generator optimization. Fine-tuning the encoder can be a later ablation.

---

# 4. Future trajectory autoencoder and latent representation

The future trajectory autoencoder maps an observed future into a compact latent representation during training and reconstructs the trajectory from that representation.

This creates the target latent space for the diffusion model.

## 4.1 Future trajectory encoder

During training, the actual future trajectory `Y_t` is available. The future encoder produces a conditional posterior:

\[
q_\chi(z_0\mid Y_t,h_t)
=
\mathcal N\left(\mu_\chi(Y_t,h_t),
\operatorname{diag}(\sigma_\chi^2(Y_t,h_t))\right).
\]

Sample using reparameterization:

\[
\boxed{z_0=\mu_\chi+\sigma_\chi\odot\xi,\qquad
\xi\sim\mathcal N(0,I).}
\]

Here, `z_0` is the clean latent representation of the future trajectory.

The future encoder is permitted to see `Y_t` during training because it constructs the diffusion target. It must not be used at inference, when the future is unknown.

## 4.2 Trajectory decoder

The trajectory decoder reconstructs the future:

\[
\hat Y_t=D_\omega(z_0,h_t).
\]

It can use temporal Transformer layers, cross-attention to historical context tokens, and cross-stock attention. Conditioning on `h_t` allows reconstruction to use both the compressed future latent and the known historical market context.

## 4.3 VAE objective

The negative evidence lower bound is:

\[
\boxed{
\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{z_0\sim q_\chi}
[-\log p_\omega(Y_t\mid z_0,h_t)]
+
\beta_{\mathrm{VAE}}
D_{KL}\left(q_\chi(z_0\mid Y_t,h_t)\Vert
\mathcal N(0,I)\right).
}
\]

For a fixed-variance Gaussian observation model, the reconstruction term is proportional to:

\[
\mathcal L_{\mathrm{recon}}
=
\|Y_t-D_\omega(z_0,h_t)\|_2^2.
\]

For a diagonal Gaussian posterior, the KL term is:

\[
D_{KL}(q_\chi\Vert\mathcal N(0,I))
=
\frac12\sum_j
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
\]

The reconstruction term preserves trajectory information; the KL term regularizes the clean latent representation.

## 4.4 Two different priors: VAE prior versus diffusion terminal prior

The VAE regularizes the clean future latent `z_0` using:

\[
p_{\mathrm{VAE}}(z_0)=\mathcal N(0,I).
\]

The diffusion process has a terminal latent `z_K`, which conventionally follows:

\[
p_{\mathrm{diff}}(z_K)=\mathcal N(0,I).
\]

These are different variables and different distributions. The proposed adaptive prior changes the distribution used to initialize reverse diffusion (`z_K`). It does not automatically change the VAE prior on `z_0`.

## 4.5 What is trained and frozen later?

Pretrain the future encoder and decoder until reconstruction is adequate. Then freeze the future encoder `Q_χ` during the initial reward-guided experiments so that the target latent coordinate system remains stable.

The trajectory decoder `D_ω` is allowed to be updated during generator optimization. Its original reconstruction objective remains active to prevent reward fine-tuning from damaging general reconstruction ability.

---

# 5. Conditional latent Diffusion Transformer

The DiT learns to transform noisy future latents into clean future latents, conditioned on the historical market representation.

## 5.1 Forward diffusion process

Let the diffusion schedule be:

\[
\beta_1,\beta_2,\ldots,\beta_K,
\]

with:

\[
\alpha_k=1-\beta_k,
\qquad
\bar\alpha_k=\prod_{j=1}^{k}\alpha_j.
\]

The forward transition is:

\[
q(z_k\mid z_{k-1})
=
\mathcal N(\sqrt{\alpha_k}z_{k-1},\beta_k I).
\]

The closed-form noising equation is:

\[
\boxed{
z_k=\sqrt{\bar\alpha_k}z_0
+\sqrt{1-\bar\alpha_k}\,\eta,
\qquad \eta\sim\mathcal N(0,I).
}
\]

At large `k`, `z_k` is mostly noise. At the terminal step, it is approximately standard Gaussian under a sufficiently good schedule.

## 5.2 Conditional DiT

The DiT receives:

1. Noisy future latent `z_k`.
2. Diffusion timestep `k`.
3. Historical market context `h_t`.

Its noise-prediction output is:

\[
\hat\eta_\theta=
\operatorname{DiT}_\theta(z_k,k,h_t).
\]

A suitable architecture can contain:

- Temporal self-attention across future latent tokens.
- Cross-stock attention across asset tokens.
- Cross-attention to historical market representations.
- Timestep embeddings.
- Adaptive normalization conditioned on time and market context.

The temporal and cross-asset attention structure allows the denoiser to model dependencies within a future market trajectory.

## 5.3 Standard diffusion objective

The epsilon-prediction loss is:

\[
\boxed{
\mathcal L_{\mathrm{diff}}
=
\mathbb E_{z_0,\eta,k}
\left[
\|\eta-\operatorname{DiT}_\theta(z_k,k,h_t)\|_2^2
\right].
}
\]

An alternative is velocity prediction:

\[
v_k=\sqrt{\bar\alpha_k}\eta
-\sqrt{1-\bar\alpha_k}z_0.
\]

The DiT can predict `v_k` instead. Use one parameterization consistently in an experiment.

## 5.4 Reverse diffusion

At inference, initialize `z_K` and repeatedly denoise it. From the predicted noise, estimate the clean latent:

\[
\hat z_0=
\frac{z_k-\sqrt{1-\bar\alpha_k}\hat\eta_\theta}
{\sqrt{\bar\alpha_k}}.
\]

For a deterministic DDIM-style update:

\[
\boxed{
z_{k-1}=
\sqrt{\bar\alpha_{k-1}}\hat z_0
+
\sqrt{1-\bar\alpha_{k-1}}\hat\eta_\theta.
}
\]

Repeat until the clean future latent has been obtained.

The number of sampling steps can be smaller than the training diffusion-step count if the selected sampler and timestep schedule support it.

## 5.5 Important distribution-matching caveat

A standard DiT trained with a Gaussian terminal distribution learns a reverse process whose initialization is matched to that training setup. Replacing the terminal Gaussian with a substantially different learned distribution is not automatically valid.

The proposed generator fine-tuning phase is intended to adapt the reverse process to samples from the new prior. However, this is a research hypothesis, not a mathematical guarantee. The generator must be evaluated on samples from both the old and new prior distributions, and the standard diffusion loss should remain active as an anchor.

A useful initial control is to restrict each prior update using a KL trust region and expose the generator to a mixture of old and new prior samples during its adaptation block.

---

# 6. Conditional terminal-noise prior

This is the main adaptive component of the proposed architecture.

## 6.1 Standard initialization

A conventional diffusion sampler begins from:

\[
z_K\sim\mathcal N(0,I).
\]

The proposed model learns a context-dependent terminal prior:

\[
\boxed{z_K\sim P_\psi(z_K\mid h_t).}
\]

The motivation is to learn a different initial-noise distribution for different historical market contexts, based on which initial samples have generated better future trajectories during training.

This is not a claim that every context necessarily needs a different prior. The adaptive prior must demonstrate out-of-sample benefit over the fixed Gaussian baseline.

## 6.2 Conditional diagonal Gaussian

For the first implementation, parameterize:

\[
P_\psi(z_K\mid h_t)
=
\mathcal N\left(
\mu_\psi(h_t),
\operatorname{diag}(\sigma_\psi^2(h_t))
\right).
\]

The prior network outputs a mean and scale:

\[
\mu_\psi(h_t),\qquad
\sigma_\psi(h_t)=\operatorname{softplus}(s_\psi(h_t))+\epsilon.
\]

Apply practical lower and upper bounds to the standard deviations. Initialize the final layers so that:

\[
\mu_\psi(h_t)\approx0,
\qquad
\sigma_\psi(h_t)\approx1.
\]

This makes the initial prior approximately standard Gaussian.

## 6.3 Reparameterized sampling

Sample using:

\[
\boxed{
z_K=\mu_\psi(h_t)
+\sigma_\psi(h_t)\odot\xi,
\qquad \xi\sim\mathcal N(0,I).
}
\]

This is differentiable with respect to the prior parameters and is also useful for pathwise-gradient experiments.

## 6.4 More expressive alternatives

A diagonal Gaussian assumes conditional independence among terminal-latent dimensions. It is a good starting point but may be restrictive.

Possible later alternatives:

- Low-rank covariance Gaussian.
- Autoregressive Gaussian prior.
- Conditional normalizing flow.
- Transformer-based conditional latent prior.

Start with the diagonal Gaussian. Increase prior expressiveness only after establishing that terminal-prior adaptation itself improves the results.

---

# 7. Candidate generation, reward evaluation, and elite selection

For a historical context `h_t`, draw `M` candidate terminal latents:

\[
z_{K,t}^{(i)}\sim P_\psi(\cdot\mid h_t),
\qquad i=1,\ldots,M.
\]

Generate trajectories:

\[
\hat Y_t^{(i)}=G_\theta(z_{K,t}^{(i)},h_t),
\]

where `G_θ` includes the complete reverse DiT process and the future trajectory decoder.

During training, the realized subsequent trajectory `Y_t` is known and can be used to score these candidates.

## 7.1 Candidate RMSE

For candidate `i`:

\[
\boxed{
d_{t,i}^{\mathrm{RMSE}}
=
\sqrt{
\frac{1}{HNF_y}
\sum_{h=1}^{H}\sum_{j=1}^{N}\sum_{f=1}^{F_y}
(\hat Y_{t,h,j,f}^{(i)}-Y_{t,h,j,f})^2}.
}
\]

Use normalized returns or appropriately scaled targets. Do not calculate error on raw prices without accounting for asset-specific scales.

A simple candidate reward is:

\[
R_{t,i}^{\mathrm{RMSE}}=-d_{t,i}^{\mathrm{RMSE}}.
\]

## 7.2 Elite selection

The recommended initial approach is top-`K_e` selection:

\[
\mathcal E_t=
\operatorname{TopK}_{i}
\left(-d_{t,i}^{\mathrm{RMSE}}\right).
\]

For an initial experiment, generate around 20 candidates and retain 4–5 elites. These are tunable experimental values.

A threshold rule is also possible:

\[
\mathcal E_t=
\{i:d_{t,i}^{\mathrm{RMSE}}\leq\tau_t\}.
\]

However, a fixed threshold may select no candidates in a difficult period or almost every candidate in an easy period. Top-`K_e` selection provides a predictable elite count.

## 7.3 Sample-level statistical rewards

RMSE can be supplemented with financial-statistics losses.

### Volatility mismatch

For asset `j`, let `v_j(Y)` denote the volatility calculated over the target horizon. Then:

\[
\mathcal L_{\mathrm{vol}}^{(i)}
=
\frac1N\sum_{j=1}^{N}
\left(v_j(\hat Y^{(i)})-v_j(Y)\right)^2.
\]

### Cross-stock correlation mismatch

Let `C(Y)` be a correlation matrix estimated from the future trajectory:

\[
\mathcal L_{\mathrm{corr}}^{(i)}
=
\frac{1}{N^2}
\|C(\hat Y^{(i)})-C(Y)\|_F^2.
\]

Short horizons can produce noisy correlation estimates. Consider shrinkage covariance or sector-level summaries.

### Temporal autocorrelation mismatch

Let `a_{j,\ell}(Y)` be the autocorrelation for asset `j` at lag `ℓ`. Then:

\[
\mathcal L_{\mathrm{ACF}}^{(i)}
=
\frac{1}{NJ}
\sum_{j=1}^{N}\sum_{\ell=1}^{J}
\left(a_{j,\ell}(\hat Y^{(i)})-a_{j,\ell}(Y)\right)^2.
\]

A sample-level reward can be:

\[
\boxed{
R_{t,i}=
-\lambda_r\mathcal L_{\mathrm{RMSE}}^{(i)}
-\lambda_v\mathcal L_{\mathrm{vol}}^{(i)}
-\lambda_c\mathcal L_{\mathrm{corr}}^{(i)}
-\lambda_a\mathcal L_{\mathrm{ACF}}^{(i)}.
}
\]

Normalize loss components before combining them. Begin with RMSE alone, then introduce other terms through controlled experiments.

## 7.4 Ensemble-level scoring rules

Some metrics assess a set of generated trajectories, not one trajectory in isolation.

### Energy Score

For `M` generated samples:

\[
\boxed{
ES=
\frac1M\sum_{i=1}^{M}\|\hat Y^{(i)}-Y\|
-
\frac{1}{2M^2}
\sum_{i=1}^{M}\sum_{j=1}^{M}
\|\hat Y^{(i)}-\hat Y^{(j)}\|.
}
\]

Lower is better. The first term measures closeness to the observation; the second rewards an appropriately dispersed ensemble.

### Variogram Score

For coordinates `a,b` indexing time–asset pairs:

\[
\boxed{
VS_p=
\sum_{a,b}w_{ab}
\left[
|Y_a-Y_b|^p
-
\mathbb E_{\hat Y}|\hat Y_a-\hat Y_b|^p
\right]^2.
}
\]

The expectation is estimated using the generated ensemble. This score assesses dependence structure across components of the multivariate trajectory.

Calculate Energy Score and Variogram Score across the candidate ensemble or a minibatch of ensemble samples. Do not incorrectly treat them as independent per-candidate rewards without defining the corresponding ensemble objective.

## 7.5 Diversity and oracle-selection caution

Selecting the lowest-RMSE samples is useful during training because the future is available as a training target. It can nevertheless cause the learned prior to concentrate around a narrow set of latent regions.

Mitigate this using:

- KL regularization against the previous prior.
- A variance floor for the conditional prior.
- Reward temperature rather than only hard selection.
- Distribution-level scoring rules.
- Fresh samples across training updates.
- Validation on complete ensembles without ground-truth candidate selection.

Do not report the best-of-`M` RMSE as the primary inference metric. At inference the future is unknown, so the model cannot select the sample that happens to be closest to it. Report ensemble mean/median point forecasts and proper probabilistic scores instead.

---

# 8. Prior optimization block: moving reference and trust region

This is the first half of the alternating training scheme.

The prior is optimized while the generator is held fixed. The reference prior is also held fixed throughout the block.

## 8.1 Outer iteration snapshots

Let `b` denote the outer iteration.

At its beginning, create detached snapshots:

\[
P_{\mathrm{ref}}^{(b)}
=P_{\bar\psi^{(b)}},
\]

and:

\[
G_{\mathrm{ref}}^{(b)}
=G_{\bar\theta^{(b)}}.
\]

These references are constant during their respective block. The online prior is initialized from the previous learned prior and is trainable.

## 8.2 Candidate generation under a frozen generator

For a batch of historical contexts:

\[
z_{K,t}^{(i)}\sim P_\psi(\cdot\mid h_t).
\]

Generate candidates using the frozen EMA generator:

\[
\hat Y_t^{(i)}=G_{\bar\theta^{(b)}}(z_{K,t}^{(i)},h_t).
\]

Evaluate rewards, select elite samples, and retain the original initial latents associated with them.

The generator is frozen because the goal of this block is to identify useful terminal-latent regions under a fixed mapping.

Fresh candidate samples should be generated as the prior updates. Repeatedly fitting only to one fixed set of elite latent vectors risks over-concentration and sample memorization.

## 8.3 Reward-weighted likelihood update

For selected elites `i ∈ E_t`, define softmax weights:

\[
\boxed{
w_{t,i}=
\frac{\exp(R_{t,i}/T)}
{\sum_{j\in\mathcal E_t}\exp(R_{t,j}/T)}.
}
\]

Here `T` is a temperature. Smaller values focus more strongly on the highest-reward candidates; larger values distribute weight more evenly.

The elite negative log-likelihood is:

\[
\mathcal L_{\mathrm{elite-prior}}
=-\frac1B\sum_{t=1}^{B}
\sum_{i\in\mathcal E_t}
 w_{t,i}\log P_\psi(z_{K,t}^{(i)}\mid h_t).
\]

This update makes the new prior assign higher likelihood to initial latents that generated better trajectories.

## 8.4 Moving-reference KL regularizer

The prior should not be permanently constrained to the original Gaussian. Instead, it should move progressively from the previous learned prior.

During outer iteration `b`, the reference is:

\[
P_{\mathrm{ref}}^{(b)}=P_{\bar\psi^{(b)}}.
\]

The conditional KL loss is:

\[
\mathcal L_{\mathrm{KL-prior}}
=
\frac1B\sum_{t=1}^{B}
D_{KL}\left(
P_\psi(\cdot\mid h_t)
\Vert
P_{\mathrm{ref}}^{(b)}(\cdot\mid h_t)
\right).
\]

The direction `D_KL(new || reference)` penalizes the new prior for assigning probability mass far outside regions supported by the previous prior.

For diagonal Gaussians:

\[
P_\psi=\mathcal N(\mu_\psi,\operatorname{diag}(\sigma_\psi^2)),
\]

\[
P_{\mathrm{ref}}=\mathcal N(\mu_{\mathrm{ref}},\operatorname{diag}(\sigma_{\mathrm{ref}}^2)),
\]

the KL divergence is:

\[
\boxed{
D_{KL}(P_\psi\Vert P_{\mathrm{ref}})
=
\frac12\sum_j\left[
\log\frac{\sigma_{\mathrm{ref},j}^2}{\sigma_{\psi,j}^2}
+
\frac{\sigma_{\psi,j}^2+(\mu_{\psi,j}-\mu_{\mathrm{ref},j})^2}
{\sigma_{\mathrm{ref},j}^2}
-1
\right].
}
\]

The reference parameters are detached and fixed throughout the prior block.

## 8.5 Trust-region constraint

The average conditional KL can be constrained:

\[
\boxed{
\frac1B\sum_{t=1}^{B}
D_{KL}(P_\psi(\cdot\mid h_t)\Vert P_{\mathrm{ref}}^{(b)}(\cdot\mid h_t))
\leq\delta.
}
\]

A Lagrangian penalty can be used:

\[
\mathcal L_{P}
=
\mathcal L_{\mathrm{elite-prior}}
+
\beta_b(\overline D_{KL}-\delta)
+
\gamma_{\mathrm{var}}\mathcal L_{\mathrm{var}}.
\]

An adaptive multiplier may be updated as:

\[
\boxed{
\beta_b\leftarrow
\beta_b\exp\left[\eta_\beta(\overline D_{KL}-\delta)\right].
}
\]

If observed KL is above the desired budget, the coefficient increases; if it is below the budget, it decreases. In practice, clamp the multiplier to a reasonable range and update it at a slower frequency than the optimizer.

The variance regularizer and explicit scale bounds prevent the prior from collapsing to extremely small variance:

\[
\sigma_{\min}\leq\sigma_\psi(h_t)\leq\sigma_{\max}.
\]

## 8.6 Complete prior objective

The complete prior loss is:

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{prior}}^{(b)}
={}&-
\frac1B\sum_{t=1}^{B}
\sum_{i\in\mathcal E_t}
 w_{t,i}\log P_\psi(z_{K,t}^{(i)}\mid h_t)
\\
&+\beta_b\frac1B\sum_{t=1}^{B}
D_{KL}\left(
P_\psi(\cdot\mid h_t)\Vert P_{\mathrm{ref}}^{(b)}(\cdot\mid h_t)
\right)
\\
&+\gamma_{\mathrm{var}}\mathcal L_{\mathrm{var}}.
\end{aligned}
}
\]

Only the online prior parameters `ψ` are optimized in this block.

## 8.7 CEM-style update versus formal policy-gradient RL

The weighted elite maximum-likelihood objective is a CEM-style reward-guided distribution update. It should not be described as a conventional multi-step RL algorithm by itself.

A formal contextual policy-gradient version is also possible. Treat `z_K` as an action sampled from policy `P_ψ(·|h_t)`, with the historical representation as the context and the generated trajectory reward as the outcome:

\[
J(\psi)=
\mathbb E_{z_K\sim P_\psi(\cdot\mid h_t)}[R(G_\theta(z_K,h_t),Y_t)]
-
\beta D_{KL}(P_\psi\Vert P_{\mathrm{ref}}).
\]

The score-function gradient is:

\[
\boxed{
\nabla_\psi J=
\mathbb E\left[
(R-b(h_t))\nabla_\psi\log P_\psi(z_K\mid h_t)
\right]
-
\beta\nabla_\psi D_{KL}.
}
\]

`b(h_t)` is a baseline for variance reduction. This is more accurately a contextual-bandit or one-step policy-optimization problem than a long-horizon RL problem, because the policy chooses the initial latent and receives a terminal trajectory reward.

Implement elite-weighted likelihood first, then compare it with policy gradients as a separate experiment.

---

# 9. Generator optimization block

After the prior block, the online prior has adapted. Update its EMA copy and construct a fixed sampling distribution for generator training.

The generator is now trainable, while the prior distributions are frozen.

## 9.1 Old and new prior snapshots

At the beginning of outer iteration `b`, the old prior reference is:

\[
P_{\mathrm{old}}^{(b)}=P_{\bar\psi^{(b)}}.
\]

After prior optimization and its EMA update, the new prior is:

\[
P_{\mathrm{new}}^{(b)}=P_{\bar\psi^{(b+1)}}.
\]

Both distributions are detached and fixed during the generator block.

## 9.2 Fixed mixture for generator training

Use:

\[
\boxed{
q_b(z_K\mid h_t)
=
\rho P_{\mathrm{new}}^{(b)}(z_K\mid h_t)
+(1-\rho)P_{\mathrm{old}}^{(b)}(z_K\mid h_t).
}
\]

This gives the generator exposure to both:

- Latent regions found useful under the previous generator.
- Newly discovered latent regions from the adapted prior.

The mixture weights and component distributions stay fixed for the entire generator optimization block. New samples can be drawn for each minibatch from this same fixed mixture.

An optional small Gaussian anchor can be added:

\[
q_b=
\rho P_{\mathrm{new}}^{(b)}
+(1-\rho-\kappa)P_{\mathrm{old}}^{(b)}
+\kappa\mathcal N(0,I),
\]

where `κ` is small and may be annealed. This is an ablation, not a mandatory permanent component.

## 9.3 Generator parameters and gradient path

Let the complete generator be:

\[
G_\theta=D_\omega\circ R_{\theta_D},
\]

where `R_{θ_D}` is the reverse DiT process and `D_ω` is the trajectory decoder. The trainable generator parameters are:

\[
\theta=(\theta_D,\omega).
\]

For generator fine-tuning, the reward gradient flows through:

\[
z_K
\rightarrow R_{\theta_D}
\rightarrow z_0
\rightarrow D_\omega
\rightarrow \hat Y
\rightarrow \mathcal L.
\]

The prior is frozen, so no prior parameter update takes place in this block.

## 9.4 Diffusion preservation loss

Keep the standard denoising objective active:

\[
\mathcal L_{\mathrm{diff}}
=
\mathbb E_{z_0,\eta,k}
\left[
\|\eta-\operatorname{DiT}_\theta(z_k,k,h_t)\|_2^2
\right].
\]

This anchors the denoiser to the original forward-diffusion learning problem and reduces the risk that reward fine-tuning destroys its learned denoising behaviour.

## 9.5 Decoder reconstruction loss

Retain the future autoencoder reconstruction objective:

\[
\boxed{
\mathcal L_{\mathrm{recon}}
=
\frac1B\sum_{t=1}^{B}
\|Y_t-D_\omega(E_\chi(Y_t,h_t),h_t)\|_2^2.
}
\]

The future encoder `E_χ` is frozen, while the decoder `D_ω` is trainable. This lets the decoder improve while retaining its ability to reconstruct actual future trajectories from the established latent coordinate system.

## 9.6 Elite reward loss

Draw candidates from the fixed mixture:

\[
z_{K,t}^{(i)}\sim q_b(\cdot\mid h_t),
\]

and generate:

\[
\hat Y_t^{(i)}=G_\theta(z_{K,t}^{(i)},h_t).
\]

Select elite samples and define:

\[
\boxed{
\mathcal L_{\mathrm{elite}}
=
\frac1B\sum_{t=1}^{B}
\sum_{i\in\mathcal E_t}
 w_{t,i}
\|\hat Y_t^{(i)}-Y_t\|_2^2.
}
\]

The elite-index selection is discrete. Gradients are nevertheless available through the generated trajectories of the selected candidates. The selection step itself is not differentiated.

For differentiable reward optimization, a deterministic DDIM-style sampler is a useful initial choice because it makes the reverse path easier to differentiate and reduces sampling variance. Stochastic sampling can still be used for evaluation and inference.

## 9.7 Distribution-level generator losses

Calculate Energy Score and Variogram Score over the complete candidate ensemble:

\[
\mathcal L_{\mathrm{distribution}}
=
\lambda_{ES}\mathcal L_{ES}
+
\lambda_{VS}\mathcal L_{VS}.
\]

These objectives help prevent the generator from focusing exclusively on one realized future and losing uncertainty or cross-asset dependence information.

They should be evaluated over all candidates (or a suitably sized ensemble), not merely the elite subset.

## 9.8 Complete generator objective

The proposed loss is:

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{generator}}
={}&
\lambda_{\mathrm{diff}}\mathcal L_{\mathrm{diff}}
+
\lambda_{\mathrm{recon}}\mathcal L_{\mathrm{recon}}
\\
&+
\lambda_{\mathrm{elite}}\mathcal L_{\mathrm{elite}}
+
\lambda_{ES}\mathcal L_{ES}
+
\lambda_{VS}\mathcal L_{VS}.
\end{aligned}
}
\]

The generator update is:

\[
\boxed{
\theta^{(b+1)}
=
\arg\min_\theta
\mathcal L_{\mathrm{generator}}(\theta;q_b).
}
\]

During this optimization, `q_b` remains fixed.

## 9.9 Training-batch composition

The generator block can use two complementary sources of training signal:

1. **Forward diffusion batches:** construct noised latent targets from actual future encodings and train the DiT with the standard diffusion loss.
2. **Reverse-generation batches:** sample terminal latents from `q_b`, run the reverse sampler, decode candidates, and calculate elite and ensemble reward losses.

These may be calculated in the same training step or on separate minibatches. In either case, keep their relative loss weights controlled and monitor the standard diffusion loss so the generator does not forget the original denoising task.

---

# 10. Alternating optimization, EMA, and snapshots

This section specifies how the two blocks interact over outer iterations.

## 10.1 Four parameter sets

Maintain:

| Parameters | Role |
|---|---|
| `ψ` | Online trainable terminal prior |
| `\bar ψ` | EMA terminal prior |
| `θ` | Online trainable DiT + trajectory decoder |
| `\bar θ` | EMA DiT + trajectory decoder |

The historical encoder and future encoder are frozen during the first reward-adaptation experiments.

## 10.2 EMA updates

Prior EMA:

\[
\boxed{
\bar\psi\leftarrow
\alpha_P\bar\psi+(1-\alpha_P)\psi.
}
\]

Generator EMA:

\[
\boxed{
\bar\theta\leftarrow
\alpha_G\bar\theta+(1-\alpha_G)\theta.
}
\]

Initial experimental values might be:

\[
\alpha_P=0.995,
\qquad
\alpha_G=0.999.
\]

These are starting values, not universal settings. Tune them using validation quality and stability diagnostics.

EMA copies are detached moving averages. They are not ordinarily trained by backpropagation. Use online weights for gradient updates and EMA weights for stable reference sampling and validation.

## 10.3 Outer iteration schedule

At the start of outer iteration `b`, snapshot:

\[
P_{\mathrm{ref}}^{(b)}=P_{\bar\psi^{(b)}},
\qquad
G_{\mathrm{ref}}^{(b)}=G_{\bar\theta^{(b)}}.
\]

### Block A: prior optimization

- Freeze `G_ref^(b)`.
- Freeze `P_ref^(b)` as the KL reference.
- Train online prior `ψ`.
- Sample candidates from the current online prior.
- Evaluate them with the frozen EMA generator.
- Select elites and optimize the prior objective.
- Update the online prior EMA during the block, but do not change the fixed reference snapshot.

At the end of the block, obtain:

\[
P_{\mathrm{new}}^{(b)}=P_{\bar\psi^{(b+1)}}.
\]

### Block B: generator optimization

Construct:

\[
q_b=
\rho P_{\mathrm{new}}^{(b)}
+(1-\rho)P_{\mathrm{old}}^{(b)}.
\]

- Freeze both prior snapshots.
- Freeze the historical encoder and future encoder initially.
- Train the online DiT and trajectory decoder.
- Keep the standard diffusion and reconstruction objectives active.
- Add reward-based elite and ensemble losses.
- Update the generator EMA during the block.

At the end of the block, start the next outer iteration with refreshed snapshots.

## 10.4 Why this is more controlled than simultaneous updates

With simultaneous updates, the prior can change the input distribution while the generator is adapting to the previous distribution. The generator can therefore chase a moving input distribution, while the prior is being evaluated using a moving generator.

Alternating optimization gives each component a block in which the other component's relevant reference is fixed:

- The prior learns useful terminal-latent regions under a fixed generator.
- The generator learns to map samples from a fixed mixture of old and new priors into better trajectories.

This does not guarantee global convergence. The joint problem remains non-convex. It is a controlled optimization strategy that should be compared empirically with simultaneous updates.

---

# 11. Full training algorithm

## 11.1 Pretraining

1. Prepare the chronological financial dataset.
2. Train the historical market encoder.
3. Train the future trajectory VAE.
4. Freeze the future trajectory encoder.
5. Train the conditional latent DiT using a standard Gaussian terminal distribution and the diffusion objective.
6. Initialize the conditional terminal prior to approximately standard Gaussian.
7. Initialize separate EMA copies of the prior and generator.

## 11.2 Alternating reward-guided training pseudocode

```python
# Conceptual pseudocode. Helper functions require implementation.

initialize_market_encoder()
initialize_future_trajectory_vae()
initialize_conditional_dit()

freeze(future_trajectory_encoder)
initialize_conditional_prior_as_standard_gaussian()

initialize_ema_prior()
initialize_ema_generator()

for outer_step in range(num_outer_steps):

    # --------------------------------------------------
    # Freeze snapshots at the beginning of the outer step
    # --------------------------------------------------
    prior_reference = frozen_copy(ema_prior)
    generator_reference = frozen_copy(ema_generator)

    # ==================================================
    # BLOCK A: PRIOR OPTIMIZATION
    # ==================================================
    freeze(generator_reference)
    freeze(prior_reference)
    train(online_prior)

    for epoch in range(prior_epochs):
        for past_data, future_truth in train_loader:

            context = market_encoder(past_data)

            # Draw fresh candidates from current online prior
            terminal_latents = online_prior.sample(
                context,
                num_samples=num_candidates
            )

            # Generator is frozen during this block
            with no_grad():
                candidates = generator_reference.generate(
                    terminal_latents, context
                )

            candidate_rmse = evaluate_rmse(candidates, future_truth)
            elite_indices = select_top_k(candidate_rmse, num_elites)

            elite_latents = terminal_latents[elite_indices]
            elite_rewards = -candidate_rmse[elite_indices]
            elite_weights = softmax(elite_rewards / temperature)

            elite_nll = weighted_negative_log_likelihood(
                online_prior,
                elite_latents,
                context,
                elite_weights
            )

            moving_kl = conditional_kl(
                online_prior,
                prior_reference,
                context
            )

            variance_penalty = prior_variance_regularization(online_prior)

            prior_loss = (
                elite_nll
                + beta * (moving_kl - kl_target)
                + gamma_var * variance_penalty
            )

            prior_optimizer.zero_grad()
            prior_loss.backward()
            clip_grad_norm_(online_prior.parameters())
            prior_optimizer.step()

            update_ema_prior()
            adapt_kl_multiplier_if_needed()

    # Snapshot the newly adapted prior after the prior block
    new_prior = frozen_copy(ema_prior)

    # ==================================================
    # BLOCK B: GENERATOR OPTIMIZATION
    # ==================================================
    freeze(prior_reference)
    freeze(new_prior)
    freeze(market_encoder)
    freeze(future_trajectory_encoder)

    train(online_dit)
    train(online_trajectory_decoder)

    # Fixed for the entire generator block
    sampling_distribution = fixed_mixture(
        new_prior,
        prior_reference,
        new_prior_weight=rho
    )

    for epoch in range(generator_epochs):
        for past_data, future_truth in train_loader:

            context = market_encoder(past_data)

            # Forward-diffusion denoising batch
            diffusion_loss = compute_diffusion_loss(
                online_dit,
                future_trajectory_encoder,
                future_truth,
                context
            )

            # Reverse-generation batch from fixed prior mixture
            terminal_latents = sampling_distribution.sample(
                context,
                num_samples=num_candidates
            )

            generated = online_generator.generate(
                terminal_latents,
                context
            )

            candidate_rmse = evaluate_rmse(generated, future_truth)
            elite_indices = select_top_k(candidate_rmse, num_elites)

            reconstruction_loss = compute_autoencoder_reconstruction_loss(
                online_trajectory_decoder,
                future_trajectory_encoder,
                future_truth,
                context
            )

            elite_loss = compute_weighted_elite_loss(
                generated,
                future_truth,
                elite_indices
            )

            energy_loss = compute_energy_score(generated, future_truth)
            variogram_loss = compute_variogram_score(generated, future_truth)

            generator_loss = (
                lambda_diff * diffusion_loss
                + lambda_recon * reconstruction_loss
                + lambda_elite * elite_loss
                + lambda_energy * energy_loss
                + lambda_variogram * variogram_loss
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            clip_grad_norm_(online_generator.parameters())
            generator_optimizer.step()

            update_ema_generator()

    # ==================================================
    # VALIDATION AND CHECKPOINTING
    # ==================================================
    validate_without_oracle_candidate_selection()
    log_prior_kl_and_entropy()
    log_ensemble_diversity()
    log_online_and_ema_metrics()
    save_checkpoint_if_validation_improves()
```

This is conceptual pseudocode, not a drop-in implementation. In particular, a production implementation must carefully handle reverse-sampler differentiability, batch and candidate dimensions, mixed precision, memory usage, and separate optimizer parameter groups.

---

# 12. Inference procedure

At inference, the subsequent ground-truth trajectory is unknown. Therefore, no reward evaluation, RMSE-based selection, or prior update takes place.

Use the final EMA models.

## Step 1: Encode recent market history

\[
h_t=E_\phi(X_t^{\mathrm{past}}).
\]

## Step 2: Sample terminal latents from the learned prior

\[
z_K^{(i)}\sim P_{\bar\psi}(z_K\mid h_t),
\qquad i=1,\ldots,M.
\]

## Step 3: Run reverse diffusion

\[
z_0^{(i)}=R_{\bar\theta_D}(z_K^{(i)},h_t).
\]

## Step 4: Decode future trajectories

\[
\boxed{
\hat Y_t^{(i)}=D_{\bar\omega}(z_0^{(i)},h_t).
}
\]

## Step 5: Convert normalized returns into prices

If the model predicts normalized log returns:

\[
\hat P_{i,t+h}
=
P_{i,t}\exp\left(
 s_{i,t}\sum_{j=1}^{h}
 \hat{\tilde r}_{i,t+j}
\right).
\]

This produces an ensemble of possible price paths for the modeled stocks.

Possible downstream analysis includes:

- Expected returns and median trajectories.
- Prediction intervals and uncertainty.
- Volatility distributions.
- Cross-stock correlations.
- Tail-risk scenarios.
- Probabilities of different market states.
- Comparisons between generated and realized market dynamics.

The learned prior is fixed during inference. It generates terminal latents conditioned on the observed historical representation.

---

# 13. Evaluation methodology

Evaluation must measure both point accuracy and the quality of the generated distribution.

## 13.1 Point forecasting accuracy

Report:

- RMSE.
- MAE.
- Directional accuracy.
- Horizon-wise RMSE/MAE.

For an ensemble of `M` trajectories, one point forecast is the ensemble mean:

\[
\bar Y_t=\frac1M\sum_{i=1}^{M}\hat Y_t^{(i)}.
\]

Calculate point-forecast metrics using `\bar Y_t` or a clearly specified median forecast.

Do not report best-of-`M` RMSE as the primary forecasting result. That metric uses the realized future to select a generated trajectory and is an oracle diagnostic, not a valid inference-time forecast.

## 13.2 Probabilistic quality

Use:

- Energy Score.
- Variogram Score.
- Marginal CRPS for individual stocks and horizons.
- Prediction interval coverage.
- Prediction interval width.
- Calibration by forecast horizon and market regime.

For example, nominal 90% prediction intervals should be checked for approximately 90% empirical coverage over sufficiently many test observations.

## 13.3 Cross-stock dependence

Compare generated and realized:

- Correlation matrices.
- Sector correlations.
- Pairwise return distributions.
- Tail dependence.
- Cross-asset lead–lag statistics.

## 13.4 Financial statistical realism

Evaluate:

| Property | Example diagnostic |
|---|---|
| Volatility clustering | ACF of squared returns |
| Heavy tails | Skewness, kurtosis, tail quantiles |
| Return autocorrelation | ACF of returns |
| Cross-asset dependence | Correlation and tail dependence |
| Drawdowns | Maximum drawdown distribution |
| Regime behaviour | Regime-conditioned statistics |

These should be evaluated across generated ensembles and multiple historical windows.

## 13.5 Optimization stability

Track:

1. Moving-reference KL divergence per outer iteration.
2. Prior entropy and predicted variance.
3. Generator diffusion loss.
4. Validation Energy Score and Variogram Score.
5. Ensemble diversity.
6. Mean reward across training windows.
7. Online-versus-EMA performance.
8. Performance on old-prior versus new-prior samples after each generator block.

A useful plot is:

\[
KL_b=
D_{KL}\left(
P_{\bar\psi^{(b)}}\Vert P_{\bar\psi^{(b-1)}}
\right),
\]

which measures prior movement between outer iterations.

Also plot prior movement against generator validation improvement. Large prior movement followed by degraded generator calibration can indicate that the prior is moving faster than the generator can adapt.

---

# 14. Ablation studies

The following experiments isolate the contribution of each proposed component.

| Experiment | Prior | Generator optimization | Purpose |
|---|---|---|---|
| A | Standard Gaussian | Standard diffusion training | Baseline |
| B | Adaptive prior | Frozen generator | Prior adaptation alone |
| C | Standard Gaussian | Reward-fine-tuned generator | Generator adaptation alone |
| D | Adaptive prior | Simultaneous updates | Test coupled optimization |
| E | Adaptive prior | Alternating, Gaussian KL reference | Test alternation |
| F | Adaptive prior | Alternating, moving prior reference | Test progressive adaptation |
| G | Adaptive prior | Moving reference + trust region + separate EMA | Full proposed method |

Additional ablations:

- No Gaussian anchor versus a small Gaussian anchor.
- Fixed KL coefficient versus adaptive trust-region multiplier.
- Different prior and generator block lengths.
- Different elite counts and reward temperatures.
- Reward-weighted MLE/CEM versus policy-gradient prior updates.
- Diagonal Gaussian versus low-rank Gaussian or conditional normalizing flow.
- RMSE-only reward versus combined sample and ensemble rewards.
- Separate EMA rates and EMA snapshot schedules.

The comparison between simultaneous and alternating updates is essential. The comparison between a fixed Gaussian reference and a moving learned-prior reference isolates the benefit of progressive prior adaptation.

---

# 15. Failure modes and mitigation

## 15.1 Prior collapse

**Failure:** The prior concentrates on a very narrow latent region, generating nearly identical trajectories.

**Mitigation:** KL trust region, variance floor, reward temperature, fresh candidate sampling, entropy monitoring, and ensemble-level scoring rules.

## 15.2 Overfitting to individual historical windows

**Failure:** The prior memorizes latent vectors that perform well on a small set of training examples.

**Mitigation:** Train across many chronological windows, draw fresh candidates after prior updates, use chronological validation, and evaluate on unseen regimes.

## 15.3 Reward exploitation

**Failure:** The generator reduces RMSE while producing unrealistic or overly smooth trajectories.

**Mitigation:** Retain diffusion and reconstruction losses, use distributional scores, evaluate financial statistical properties, and monitor ensemble diversity.

## 15.4 Prior–generator oscillation

**Failure:** The prior changes faster than the generator adapts, producing alternating degradation and recovery.

**Mitigation:** Moving-reference KL trust region, separate EMA snapshots, conservative learning rates, alternating blocks, and validation-based block stopping.

## 15.5 Decoder degradation

**Failure:** The decoder specializes to elite trajectories and loses general reconstruction capability.

**Mitigation:** Retain VAE reconstruction loss, freeze the future encoder, use a conservative decoder learning rate, and evaluate on held-out future latents.

## 15.6 Terminal-prior mismatch

**Failure:** The learned terminal prior samples regions that the pretrained DiT has not learned to denoise.

**Mitigation:** Limit each prior update with KL trust region, train the generator on a mixture of old and new priors, retain the standard diffusion loss, and measure performance separately on old-prior and new-prior samples.

This is a central scientific risk of the project. A learned terminal prior is useful only if the adapted reverse process can reliably transform its samples into valid future trajectories.

## 15.7 Noisy reward estimates

**Failure:** Short future horizons make volatility and correlation reward estimates unreliable.

**Mitigation:** Begin with RMSE, use longer horizons for statistical rewards, use shrinkage covariance estimates, aggregate across windows, and normalize reward components.

---

# 16. Suggested implementation roadmap

## Stage 1: Data pipeline and simple baselines

- Construct the financial universe and historical data.
- Create past/future rolling windows.
- Implement adjusted-return processing and scaling.
- Create chronological splits and leakage checks.
- Establish simple forecasting baselines.

**Deliverable:** Reliable multivariate financial training and evaluation dataset.

## Stage 2: Historical market encoder

- Implement the temporal Transformer.
- Construct historical stock relationship graphs.
- Implement graph aggregation.
- Add global market-context pooling.
- Evaluate representation quality.

**Deliverable:** Historical context representation `h_t`.

## Stage 3: Future trajectory autoencoder

- Implement future trajectory encoder.
- Implement trajectory decoder.
- Train VAE reconstruction objective.
- Evaluate reconstruction and latent stability.
- Freeze the future encoder for initial diffusion experiments.

**Deliverable:** Stable future latent space and decoder.

## Stage 4: Conditional latent DiT

- Implement forward diffusion.
- Implement conditional DiT.
- Train epsilon or velocity prediction.
- Use a standard Gaussian terminal distribution.
- Evaluate generated trajectory distribution.

**Deliverable:** Functional baseline latent diffusion forecaster.

## Stage 5: Adaptive terminal prior

- Initialize conditional Gaussian prior near standard normal.
- Freeze EMA generator.
- Generate multiple candidates.
- Calculate RMSE.
- Select elites.
- Train prior using weighted likelihood and moving-reference KL.

**Deliverable:** Reward-adaptive terminal prior.

## Stage 6: Alternating generator refinement

- Construct fixed mixture of previous and new prior snapshots.
- Freeze prior snapshots.
- Fine-tune DiT and decoder.
- Retain diffusion and reconstruction objectives.
- Add elite and ensemble losses.
- Update generator EMA.

**Deliverable:** Complete alternating prior–generator optimization.

## Stage 7: Advanced rewards and ablations

- Introduce volatility and dependence objectives.
- Add Energy Score and Variogram Score.
- Compare CEM-style updates with policy gradients.
- Compare simultaneous and alternating optimization.
- Compare fixed and moving KL references.
- Measure calibration and out-of-distribution performance.

**Deliverable:** Controlled research results and ablation studies.

## Stage 8: Analysis interface

Build an interface that allows a researcher to inspect:

- Multiple generated future trajectories.
- Prediction intervals and uncertainty.
- Cross-stock correlation scenarios.
- Tail-risk scenarios.
- Regime-conditioned forecasts.
- Historical periods that resemble generated scenarios.

**Deliverable:** Interactive financial market research and scenario-analysis system.

---

# 17. Initial experimental configuration

These values are starting points for experimentation, not optimized settings.

| Parameter | Initial configuration |
|---|---|
| Market | Indian equities |
| Number of stocks | 30–50 |
| Lookback | 60 trading days |
| Forecast horizon | 20 trading days |
| Historical features | 10–15 per stock |
| Future target | Normalized log returns |
| Historical encoder | Temporal Transformer + Graph Encoder |
| Future representation | VAE |
| Latent dimension | 64–128 |
| Denoiser | Conditional DiT |
| Diffusion training steps | 100–500 initially |
| Terminal prior | Conditional diagonal Gaussian |
| Candidate trajectories | 20 initially |
| Elite trajectories | 4–5 initially |
| Prior block | 10 epochs initially |
| Generator block | 10 epochs initially |
| Prior update | Reward-weighted MLE / CEM initially |
| Prior regularization | KL to previous EMA prior |
| Generator update | Differentiable reward fine-tuning |
| Prior EMA decay | 0.995 initially |
| Generator EMA decay | 0.999 initially |

The number of epochs per block should be ablated. A fixed 10/10 schedule is a starting design, not a requirement. Validation-based stopping or adaptive block lengths may be preferable.

---

# 18. Research contributions and hypotheses

The individual ingredients—conditional latent diffusion, DiT, reward-guided diffusion optimization, initial-noise adaptation, elite-sample optimization, alternating updates, and EMA—have each appeared in related machine-learning research. The project should not claim novelty simply from combining them.

The contribution should be formulated around their interaction for multivariate financial trajectory generation.

## Hypothesis 1: Adaptive financial terminal prior

A context-dependent terminal prior can improve the quality of generated future trajectories compared with a fixed Gaussian terminal distribution.

## Hypothesis 2: Progressive prior–generator co-adaptation

Progressively adapting the terminal prior while periodically refining the generator can improve the generator's ability to use useful latent regions without causing the instability associated with simultaneous updates.

## Hypothesis 3: Moving-reference trust-region optimization

A moving learned-prior KL reference, combined with trust-region control and separate EMA snapshots, can produce more stable prior updates than either an unrestricted prior or a prior permanently anchored to the original Gaussian.

## Hypothesis 4: Distributional quality under market shifts

Reward-guided adaptation can improve probabilistic forecast quality while preserving diversity, calibration, and cross-asset dependence, including in previously unseen market regimes.

A particularly interesting analysis is whether the learned prior parameters vary systematically with market representations and whether these changes generalize across unseen historical periods.

---

# 19. Final architecture summary

The full model can be written compactly as follows.

Historical representation:

\[
h_t=E_\phi(X_t^{\mathrm{past}}).
\]

Training-only future latent:

\[
z_0\sim q_\chi(z_0\mid Y_t,h_t).
\]

Diffusion forward process:

\[
z_k=\sqrt{\bar\alpha_k}z_0
+\sqrt{1-\bar\alpha_k}\eta.
\]

Conditional terminal prior:

\[
z_K\sim P_\psi(z_K\mid h_t).
\]

Reverse DiT and trajectory decoder:

\[
\boxed{
\hat Y_t=
D_\omega(R_{\theta_D}(z_K,h_t),h_t).
}
\]

The training process alternates between:

**Prior block:**

\[
\boxed{
\begin{aligned}
\min_\psi\quad
&-\mathbb E_{\mathrm{elite}}[\log P_\psi(z_K\mid h_t)]
\\
&+\beta_bD_{KL}\left(
P_\psi(\cdot\mid h_t)
\Vert P_{\bar\psi^{(b)}}(\cdot\mid h_t)
\right)
+\gamma_{\mathrm{var}}\mathcal L_{\mathrm{var}}.
\end{aligned}
}
\]

**Generator block:**

First define the fixed sampling distribution:

\[
q_b=
\rho P_{\bar\psi^{(b+1)}}
+(1-\rho)P_{\bar\psi^{(b)}}.
\]

Then optimize:

\[
\boxed{
\begin{aligned}
\min_\theta\quad
\mathbb E_{z_K\sim q_b}\big[
&\lambda_{\mathrm{diff}}\mathcal L_{\mathrm{diff}}
+\lambda_{\mathrm{recon}}\mathcal L_{\mathrm{recon}}
\\
&+\lambda_{\mathrm{elite}}\mathcal L_{\mathrm{elite}}
+\lambda_{ES}\mathcal L_{ES}
+\lambda_{VS}\mathcal L_{VS}
\big].
\end{aligned}
}
\]

Separate EMA networks are updated after their corresponding online updates. The next outer iteration uses the updated EMA prior as the new moving reference and the updated EMA generator as the fixed generator reference for the prior block.

---

# 20. Final recommendations

The recommended first implementation is deliberately conservative:

1. Use 30–50 stocks and daily normalized returns.
2. Build a temporal-plus-graph historical encoder.
3. Train a future trajectory VAE and freeze its future encoder.
4. Train a conditional latent DiT with a standard Gaussian terminal distribution.
5. Introduce a conditional diagonal-Gaussian terminal prior initialized near standard normal.
6. Use RMSE-based top-`K` elite selection and reward-weighted prior likelihood updates.
7. Regularize each new prior against the previous EMA prior using a KL trust region.
8. Fine-tune the DiT and trajectory decoder in a separate block using a fixed mixture of old and new prior snapshots.
9. Retain the diffusion and decoder reconstruction objectives while adding elite and ensemble reward losses.
10. Maintain separate EMA copies and evaluate complete ensembles without ground-truth-based candidate selection.

The highest-risk technical assumption is that a changed terminal prior can be paired with a reverse process that was originally trained under a Gaussian terminal distribution. The alternating generator-refinement block is intended to address this mismatch, but the experiment must directly test it. Monitor performance on old-prior and new-prior samples separately and retain standard diffusion training as an anchor.

The most important research result will not simply be a lower training RMSE. It will be evidence that the learned prior and adapted generator produce better calibrated, diverse, statistically realistic multivariate financial trajectories on strictly unseen future periods.

**Proposed project title:**

> **Progressive Reward-Guided Prior Adaptation with Alternating Generator Refinement for Latent Financial Diffusion Models**

**Central research question:**

> Can a conditional diffusion terminal prior be progressively optimized using trajectory-level reward feedback, while alternating generator refinement with moving-reference trust-region regularization, to improve probabilistic multivariate financial forecasting without sacrificing calibration, diversity, and cross-asset dependence?
