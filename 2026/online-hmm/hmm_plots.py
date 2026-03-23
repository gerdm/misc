"""
Produce all plots for the online Hidden Markov Model demo.
"""
import os
import jax
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

from hmm_utils import (
    BeliefRegimes, HMMDetector, Cfg, ParticleState,
    make_transition_matrix, make_step_fn,
    update_gauss, update_conditional, make_regime_step,
    make_streaming_step, build_weights,
)

# ---------------------------------------------------------------------------
# Plot config
# ---------------------------------------------------------------------------
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (7, 3.0)
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
sns.set_palette("colorblind")
colors = sns.color_palette()

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(314)
key_sample, key_run = jax.random.split(key)

n_regimes = 3
obs_var = 1.0
means = jnp.array([-1, 0, 1]) * 2.0
obs_std = jnp.sqrt(obs_var)

transition_matrix = make_transition_matrix(n_regimes, diag_val=0.99)
log_transition_matrix = jnp.log(transition_matrix)

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
n_steps = 1500
timesteps = jnp.arange(n_steps)
state_init = 0

step_fn = make_step_fn(transition_matrix, means, obs_var)
keys = jax.random.split(key_sample, n_steps)
_, (states, obs) = jax.lax.scan(step_fn, state_init, keys)

regime_changes_ix = jnp.where(jnp.diff(states) != 0)[0]
regime_changes_ix = jnp.insert(regime_changes_ix, 0, 0)
regime_changes_ix = jnp.append(regime_changes_ix, n_steps - 1)

# ---------------------------------------------------------------------------
# Plot 1: HMM sample
# ---------------------------------------------------------------------------
plt.figure()
plt.plot(obs, c="black", label="$y_t$")
plt.plot(means.at[states].get(), c="darkorange", linewidth=2, label=r"$\theta_{z_t}$")
plt.xlabel("timestep ($t$)")
plt.ylabel("observation space")
plt.title("Gaussian HMM | $K=3$")
plt.legend()
plt.ylim(-5, 5)
plt.grid(alpha=0.3)
plt.savefig("figures/hmm-sample-k3.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Known-regime posteriors
# ---------------------------------------------------------------------------
bel_init = BeliefRegimes(means=jnp.zeros(n_regimes), variances=jnp.ones(n_regimes) * 0.5)
_, bel_hist = jax.lax.scan(partial(update_conditional, sigma2=obs_var), bel_init, (obs, states))

ubounds = bel_hist.means + 2 * jnp.sqrt(bel_hist.variances)
lbounds = bel_hist.means - 2 * jnp.sqrt(bel_hist.variances)

# Plot 2: posterior estimate (known regimes)
plt.figure()
plt.plot(bel_hist.means, linewidth=2)
for k in range(n_regimes):
    plt.fill_between(timesteps, lbounds[:, k], ubounds[:, k], alpha=0.3)
for iinit, iend in zip(regime_changes_ix[:-1], regime_changes_ix[1:]):
    plt.axvspan(iinit, iend, alpha=0.3, linewidth=1, hatch="|",
                edgecolor=colors[states[iend]], facecolor="none", zorder=0)
for mean in means:
    plt.axhline(y=mean, c="black", linestyle="--", alpha=0.5)
plt.text(50, -1.5, r"$p(\theta_1 \mid {\cal Y}_{t,1})$", color=colors[0])
plt.text(120, 2.5, r"$p(\theta_3 \mid {\cal Y}_{t,3})$", color=colors[2])
plt.text(180, 0.5, r"$p(\theta_2 \mid {\cal Y}_{t,2})$", color=colors[1])
plt.xlim(0, 230)
plt.xlabel("timestep ($t$)")
plt.ylabel("observation ($y_t$)")
plt.title("Posterior estimate (known regimes)")
plt.savefig("figures/hmm-posterior-estimate-known-regime.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Known-parameter regime detection
# ---------------------------------------------------------------------------
S = 10
key_init, key_eval = jax.random.split(key_run)
bel_detector = HMMDetector(
    regimes=jax.random.choice(key_init, n_regimes, (S,)),
    log_weights=jnp.zeros(S),
)

regime_step = make_regime_step(log_transition_matrix, means, obs_std, S)
_, (hist_regimes, hist_log_weights) = jax.lax.scan(regime_step, bel_detector, obs)
e_mean = (means.at[hist_regimes].get() * jnp.exp(hist_log_weights)).sum(axis=1)

# Plot 3: mean estimate (known parameters)
plt.figure()
plt.plot(obs, c="black", label="observations")
plt.plot(e_mean, c="crimson", linewidth=2, label=f"est. mean | S={S}")
plt.legend()
plt.xlabel("timestep")
plt.title("Mean estimate (known parameters)")
plt.grid(alpha=0.3)
plt.savefig("figures/hmm-mean-estimate-known-params.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Streaming HMM (unknown parameters)
# ---------------------------------------------------------------------------
n_particles = 5
mean_init = jnp.array([0.0])
var_init = jnp.array([1.0])

cfg = Cfg(var_obs=obs_var, num_particles=n_particles, num_regimes=n_regimes)
bel_particle = ParticleState.init(key_init, mean_init, var_init, n_particles, n_regimes, n_steps)

streaming_step = make_streaming_step(log_transition_matrix, transition_matrix, cfg)
_step = partial(streaming_step, cfg=cfg)
bel_final, (hist_lw, hist_mean, hist_variance, hist_forecast) = jax.lax.scan(_step, bel_particle, obs)
hist_weights = build_weights(hist_lw)

# Posterior mean estimate per regime
mean_est = jnp.einsum("ts,tsk...->tk", hist_weights, hist_mean)

# Plot 4: regime mean estimates
plt.figure()
plt.plot(mean_est[:, [2, 0, 1]], linewidth=3)
for iinit, iend in zip(regime_changes_ix[:-1], regime_changes_ix[1:]):
    plt.axvspan(iinit, iend, color=colors[states[iend]], alpha=0.2, linewidth=1)
for mean in means:
    plt.axhline(y=mean, c="black", linestyle="--", alpha=0.5)
plt.ylabel("mean estimate")
plt.xlabel("timestep")
plt.xlim(0, 300)
plt.grid(alpha=0.3)
plt.savefig("figures/hmm-mean-estimate.png", dpi=300, bbox_inches="tight")
plt.close()

# Particle-weighted filtered estimate
hist_regimes_ohe = jax.nn.one_hot(bel_final.regime, n_regimes, axis=0)
proba_regimes = jnp.einsum("kst,ts->tk", hist_regimes_ohe[[2, 0, 1], ...], hist_weights)
pmean_est = jnp.einsum("tsk...,kst,ts->t", hist_mean, hist_regimes_ohe, hist_weights)

# Plot 5: filtered estimate (observations and posterior estimates stacked)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

plt.sca(axs[0])
plt.scatter(timesteps, obs, c="black", label="$y_t$", s=3)
plt.plot(means.at[states].get(), c="darkorange", linewidth=2, label=r"$\theta_{z_t}$")
plt.ylabel("$y_t$")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-3, 3)
plt.xlim(0, 230)

plt.sca(axs[1])
ubounds = bel_hist.means + 2 * jnp.sqrt(bel_hist.variances)
lbounds = bel_hist.means - 2 * jnp.sqrt(bel_hist.variances)
plt.plot(bel_hist.means, linewidth=2)

for k in range(n_regimes):
    plt.fill_between(timesteps, lbounds[:, k], ubounds[:, k], alpha=0.3)

for iinit, iend in zip(regime_changes_ix[:-1], regime_changes_ix[1:]):
    plt.axvspan(iinit, iend, alpha=0.3, linewidth=1, hatch="|",
                edgecolor=colors[states[iend]], facecolor="none", zorder=0)

for mean in means:
    plt.axhline(y=mean, c="black", linestyle="--", alpha=0.5)

plt.text(50, -1.5, r"$p(\theta_1 \mid {\cal Y}_{t,1})$", color=colors[0])
plt.text(120, 2.5, r"$p(\theta_3 \mid {\cal Y}_{t,3})$", color=colors[2])
plt.text(180, 0.5, r"$p(\theta_2 \mid {\cal Y}_{t,2})$", color=colors[1])

plt.xlim(0, 230)
plt.ylim(-3, 3)
plt.xlabel("timestep ($t$)")
plt.ylabel("Estimates")

plt.suptitle("Gaussian HMM | $K=3$")
plt.savefig("figures/hmm-estimation.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 6: forecast with uncertainty band
lbound = hist_forecast["mean"] - hist_forecast["stdev_obs"]
ubound = hist_forecast["mean"] + hist_forecast["stdev_obs"]

plt.figure()
plt.fill_between(timesteps, lbound, ubound, color="crimson", alpha=0.4, linewidth=0)
plt.plot(obs, c="gray", alpha=0.6, label="$y_t$", zorder=0)
plt.plot(hist_forecast["mean"], c="crimson", linewidth=2,
         label=r"$\mathbb{E}[y_{t} \mid {\cal Y}_{t-1}]$")
plt.plot(means.at[states].get(), c="black", zorder=1, linewidth=2, label=r"$\theta_{z_t}$")
plt.legend(ncol=3, loc="lower right")
plt.ylim(-4.5, 4.5)
plt.grid(alpha=0.3)
plt.xlabel("timestep ($t$)")
plt.savefig("figures/hmm-forecast.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot 7: forecast slice (zoomed)
plt.figure()
plt.fill_between(timesteps, lbound, ubound, color="crimson", alpha=0.3, linewidth=0)
plt.plot(obs, c="gray", alpha=0.7, label="$y_t$ — obs", zorder=0)
plt.plot(hist_forecast["mean"], c="crimson", linewidth=2,
         label=r"$\mathbb{E}[y_{t} \mid {\cal Y}_{t-1}]$ — forecast",
         marker="o", markersize=5)
plt.plot(means.at[states].get(), c="black", zorder=1, linewidth=2, label=r"$\theta_t$ — true")
plt.legend()
plt.xlim(440, 490)
plt.ylim(-4, 4)
plt.grid(alpha=0.3)
plt.xlabel("timestep ($t$)")
plt.savefig("figures/hmm-estimation-slice-clean.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Classical Bayesian baseline (assumes single mean, but DGP is HMM)
# ---------------------------------------------------------------------------

# Use the same HMM-generated data but run classical Bayesian updates
# that assume a single unknown mean (no regime awareness)
mean_prior = 0.0
var_prior = 1.0

def update_baseline_step(carry, obs_t):
    mean, var = carry
    mean_upd, var_upd = update_gauss(obs_t, mean, var, obs_var)
    return (mean_upd, var_upd), (mean_upd, var_upd)

(mean_final, var_final), (means_baseline, vars_baseline) = jax.lax.scan(
    update_baseline_step,
    (mean_prior, var_prior),
    obs
)

# Plot 8: Classical Bayesian baseline (assumes single mean)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

plt.sca(axs[0])
plt.scatter(timesteps, obs, c="black", label="$y_t$", s=3)
plt.plot(means.at[states].get(), c="darkorange", linewidth=2, label=r"$\theta_{z_t}$ (true)")
plt.ylabel("$y_t$")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-3, 3)
plt.xlim(0, 230)

plt.sca(axs[1])
ubounds_baseline = means_baseline + 2 * jnp.sqrt(vars_baseline)
lbounds_baseline = means_baseline - 2 * jnp.sqrt(vars_baseline)
plt.plot(means_baseline, linewidth=2, label=r"$\mathbb{E}[\theta \mid \mathcal{Y}_t]$ (no regimes)")
plt.fill_between(timesteps, lbounds_baseline, ubounds_baseline, alpha=0.3)

for iinit, iend in zip(regime_changes_ix[:-1], regime_changes_ix[1:]):
    plt.axvspan(iinit, iend, alpha=0.3, linewidth=1, hatch="|",
                edgecolor=colors[states[iend]], facecolor="none", zorder=0)

plt.xlim(0, 230)
plt.ylim(-3, 3)
plt.xlabel("timestep ($t$)")
plt.ylabel("Estimates")
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle("Classical Bayesian Baseline (DGP is HMM, model assumes single mean)")
plt.savefig("figures/bayesian-baseline.png", dpi=300, bbox_inches="tight")
plt.close()

print("All figures saved to ./figures/")
