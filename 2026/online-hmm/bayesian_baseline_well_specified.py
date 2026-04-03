"""
Classical Bayesian baseline in a well-specified setting:
observations come from a single Gaussian with fixed mean.
"""
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from hmm_utils import update_gauss


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
# Setup and data generation (well-specified single Gaussian DGP)
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(314)

n_steps = 1500
timesteps = jnp.arange(n_steps)
obs_var = 1.0
obs_std = jnp.sqrt(obs_var)

true_mean = 0.75
obs = true_mean + jax.random.normal(key, shape=(n_steps,)) * obs_std

# ---------------------------------------------------------------------------
# Classical Bayesian updates for a single unknown mean
# ---------------------------------------------------------------------------
mean_prior = 0.0
var_prior = 1.0


def update_baseline_step(carry, obs_t):
    mean, var = carry
    mean_upd, var_upd = update_gauss(obs_t, mean, var, obs_var)
    return (mean_upd, var_upd), (mean_upd, var_upd)


(_, _), (means_baseline, vars_baseline) = jax.lax.scan(
    update_baseline_step,
    (mean_prior, var_prior),
    obs,
)

# ---------------------------------------------------------------------------
# Plot: Classical Bayesian baseline in well-specified setting
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

plt.sca(axs[0])
plt.scatter(timesteps, obs, c="black", label="$y_t$", s=3)
plt.axhline(y=true_mean, c="darkorange", linewidth=2, label=r"$\theta$ (true)")
plt.ylabel("$y_t$")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-3, 3)
plt.xlim(0, 230)

plt.sca(axs[1])
ubounds_baseline = means_baseline + 2 * jnp.sqrt(vars_baseline)
lbounds_baseline = means_baseline - 2 * jnp.sqrt(vars_baseline)
plt.plot(means_baseline, linewidth=2, color=colors[0], label=r"$\mathbb{E}[\theta \mid \mathcal{Y}_t]$")
plt.fill_between(timesteps, lbounds_baseline, ubounds_baseline, alpha=0.3, color=colors[0])
plt.axhline(y=true_mean, c="darkorange", linewidth=2, linestyle="--", label=r"$\theta$ (true)")

plt.xlim(0, 230)
plt.ylim(-3, 3)
plt.xlabel("timestep ($t$)")
plt.ylabel("Estimates")
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle("Classical Bayesian Baseline (Well-specified single-Gaussian DGP)")
plt.savefig("figures/bayesian-baseline-well-specified.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure to ./figures/bayesian-baseline-well-specified.png")
