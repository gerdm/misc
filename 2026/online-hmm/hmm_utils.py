"""
Utility functions for online Hidden Markov Model with Gaussian observations.
"""
import jax
import chex
import einops
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@chex.dataclass
class BeliefRegimes:
    means: chex.Array
    variances: chex.Array


@chex.dataclass
class HMMDetector:
    regimes: chex.Array
    log_weights: chex.Array


@chex.dataclass
class Cfg:
    var_obs: float
    num_particles: int
    num_regimes: int


@chex.dataclass
class ParticleState:
    means: jax.Array
    variances: jax.Array
    regime: jax.Array
    log_weight: jax.Array
    timestep: jax.Array

    @staticmethod
    def init(key, mean, cov, n_particles, n_regimes, n_steps):
        key_mean, _ = jax.random.split(key)
        means = jax.random.normal(key_mean, (n_particles, n_regimes, 1)) * jnp.sqrt(cov)
        variances = einops.repeat(cov, "i -> s k i", s=n_particles, k=n_regimes)
        log_weights = jnp.full(n_particles, -jnp.log(n_particles))
        timestep = jnp.zeros(n_particles)
        regimes = jnp.zeros((n_particles, n_steps)).astype(int)
        return ParticleState(
            means=means,
            variances=variances,
            regime=regimes,
            log_weight=log_weights,
            timestep=timestep,
        )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_transition_matrix(n_regimes, diag_val=0.99):
    off = (1 - diag_val) / (n_regimes - 1)
    T = jnp.full((n_regimes, n_regimes), off)
    T = T.at[jnp.diag_indices(n_regimes)].set(diag_val)
    return T


def make_step_fn(transition_matrix, means, obs_var):
    def step(state, key):
        key_state, key_noise = jax.random.split(key)
        proba_change = transition_matrix[state]
        state_next = jax.random.choice(key_state, len(proba_change), p=proba_change)
        err = jax.random.normal(key_noise) * jnp.sqrt(obs_var)
        obs = means[state_next] + err
        return state_next, (state_next, obs)
    return step


# ---------------------------------------------------------------------------
# Known-regime updates
# ---------------------------------------------------------------------------

def update_gauss(y, mean, var, sigma2):
    err = y - mean
    kt = var / (var + sigma2)
    mean_update = mean + kt * err
    var_update = (1 - kt) * var
    return mean_update, var_update


def update_conditional(bel, xs, sigma2):
    y, z = xs
    mean = bel.means[z]
    var = bel.variances[z]
    mean_update, var_update = update_gauss(y, mean, var, sigma2)
    bel_update = bel.replace(
        means=bel.means.at[z].set(mean_update),
        variances=bel.variances.at[z].set(var_update),
    )
    return bel_update, bel


# ---------------------------------------------------------------------------
# Known-parameter regime detector (discrete beam search)
# ---------------------------------------------------------------------------

def make_regime_step(log_transition_matrix, means, obs_std, S):
    n_regimes = len(means)

    @partial(jax.vmap, in_axes=(0, 0, None))
    def update_log_weight(regime, log_weight, obs):
        log_transition = log_transition_matrix[regime]
        log_likelihoods = jax.scipy.stats.norm.logpdf(obs, means, obs_std)
        return log_weight + log_transition + log_likelihoods

    def step(bel, obs):
        log_weights_update = update_log_weight(bel.regimes, bel.log_weights, obs).ravel()
        ixs = jnp.argsort(log_weights_update, descending=True)[:S]
        regimes_new = ixs % n_regimes
        log_weights_new = log_weights_update[ixs]
        log_weights_new = log_weights_new - jax.nn.logsumexp(log_weights_new)
        bel = bel.replace(regimes=regimes_new, log_weights=log_weights_new)
        return bel, (regimes_new, log_weights_new)

    return step


# ---------------------------------------------------------------------------
# Streaming HMM (unknown parameters)
# ---------------------------------------------------------------------------

def flatten_fn(fn):
    def flatten_particles(tree):
        res = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), tree)
        return res
    return lambda *x: flatten_particles(fn(*x))


def make_streaming_step(log_transition_matrix, transition_matrix, cfg):

    @flatten_fn
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def update_conditional_posterior(y, regime, bel, cfg):
        yhat = mean = bel.means[regime]
        var = bel.variances[regime]
        mean_update, var_update = update_gauss(y, mean, var, cfg.var_obs)
        pred_sttdev = jnp.sqrt(var + cfg.var_obs)
        log_pp = jax.scipy.stats.norm.logpdf(y, yhat, pred_sttdev).squeeze()
        regime_curr = bel.regime[bel.timestep.astype(int)]
        log_p_transition = log_transition_matrix[regime_curr, regime]
        timestep_new = bel.timestep + 1
        bel = bel.replace(
            means=bel.means.at[regime].set(mean_update),
            variances=bel.variances.at[regime].set(var_update),
            regime=bel.regime.at[timestep_new.astype(int)].set(regime),
            timestep=timestep_new,
        )
        return bel, log_pp, log_p_transition

    @jax.vmap
    def update_log_weights(bel, log_pp, log_p_transition):
        log_weight = log_pp + log_p_transition + bel.log_weight
        return bel.replace(log_weight=log_weight)

    def beam_search(bel, K):
        log_weights = bel.log_weight
        indices = jnp.argsort(log_weights, descending=True)[:K]
        bel = jax.tree.map(lambda x: x[indices], bel)
        log_weights = log_weights[indices]
        log_weights = log_weights - jax.nn.logsumexp(log_weights)
        return bel.replace(log_weight=log_weights)

    def forecast(bel, cfg):
        means_bel = bel.means
        variances = bel.variances
        weights = jnp.exp(bel.log_weight - jax.nn.logsumexp(bel.log_weight))
        timestep = bel.timestep.astype(int)[0]
        regime = bel.regime[:, timestep]
        p_transition = transition_matrix.at[regime].get()
        yhat = jnp.einsum("s,sk,sk...->", weights, p_transition, means_bel)
        mean2 = jnp.einsum("s,sk,sk...->", weights, p_transition, means_bel ** 2 + variances)
        yhat2 = mean2 + cfg.var_obs
        yhat_std = jnp.sqrt(yhat2 - yhat ** 2)
        mean_std = jnp.sqrt(mean2 - yhat ** 2)
        return {
            "mean": yhat,
            "stdev_obs": yhat_std,
            "stdev_param": mean_std,
        }

    def step(bel, y, cfg):
        fcst = forecast(bel, cfg)
        regimes = jnp.arange(cfg.num_regimes)
        bel_update, log_pp, log_p_transition = update_conditional_posterior(y, regimes, bel, cfg)
        bel_update = update_log_weights(bel_update, log_pp, log_p_transition)
        bel_update = beam_search(bel_update, cfg.num_particles)
        return bel_update, (bel_update.log_weight, bel_update.means, bel_update.variances, fcst)

    return step


@jax.vmap
def build_weights(log_weights):
    log_weights_norm = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(log_weights_norm)
