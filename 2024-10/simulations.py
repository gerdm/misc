import jax
import chex
import jax.numpy as jnp
from functools import partial

@chex.dataclass
class ParamsLatent:
    alpha: float
    beta: float
    delta: float
    gamma: float
    dt: float
    noise: float


@chex.dataclass
class ParamsObserved:
    noise: float


def latent_step(key, z_prev, params):
    z1, z2 = z_prev
    noise = jax.random.normal(key) * params.noise * jnp.sqrt(params.dt)
    
    z1_next = params.alpha * z1 - params.beta * z1 * z2
    z2_next = params.delta * z1 * z2 - params.gamma * z2
    z_next = jnp.array([z1_next, z2_next])
    
    z_next = z_prev + params.dt * z_next + noise
    
    return z_next


def observed_step(key, z_val, params):
    """
    Observed step is simply the current latent
    observed value + gaussian noise
    """
    noise = jax.random.normal(key, (2,)) * params.noise
    obs = z_val + noise
    return obs


def step_latent_observed(z_prev, key, latent_fn, observed_fn,
                             params_latent, params_obs):
    key_latent, key_obs = jax.random.split(key)
    
    z_next = latent_fn(key_latent, z_prev, params_latent)
    x_next = observed_step(key_obs, z_next, params_obs)
    
    res = {
        "latent": z_next,
        "observed": x_next
    }
    return z_next, res
    

def simulate_lotka_volterra(
    key, z0, n_steps, params_latent, params_obs
):
    key_init, key_sample = jax.random.split(key)
    z0 = z0 + jax.random.uniform(key_init, (2,), minval=-0.2, maxval=0.2)
    
    keys = jax.random.split(key_sample, n_steps)
    part_step_latent_obs = partial(step_latent_observed,
                                   latent_fn=latent_step,
                                   observed_fn=observed_step,
                                   params_latent=params_latent,
                                   params_obs=params_obs)
    
    _, hist = jax.lax.scan(part_step_latent_obs, z0, keys)
    return hist


@partial(jax.vmap, in_axes=(0, None, None, None, None), out_axes=-1)
def multiple_simulate_lotka_volterra(
    key, z0, n_steps, params_latent, params_obs
):
    return simulate_lotka_volterra(
        key, z0, n_steps, params_latent, params_obs
    )