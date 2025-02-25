from typing import NamedTuple

from flax import linen as nn
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfp


def forager_network(x, hidden_units):
    kernel_initializer = jax.nn.initializers.orthogonal(jnp.sqrt(2))
    bias_initializer = jax.nn.initializers.constant(0)

    x = nn.Conv(
        features=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=kernel_initializer)(x)

    x = nn.relu(x)
    x = x.reshape((-1))

    x = nn.Dense(features=hidden_units, kernel_init=kernel_initializer, bias_init=bias_initializer)(x)
    x = nn.relu(x)
    return x

class SacActorOutput(NamedTuple):
    """The output of a SAC actor."""
    action: jnp.ndarray
    log_prob: jnp.ndarray


class SacCriticOutput(NamedTuple):
    """The output of a SAC critic."""
    q_value1: jnp.ndarray
    q_value2: jnp.ndarray

class SacMemoryActorOutput(NamedTuple):
    """The output of a SAC actor memory."""
    action: jnp.ndarray
    log_prob: jnp.ndarray
    carry: jnp.ndarray


class SacMemoryCriticOutput(NamedTuple):
    """The output of a SAC critic memory."""
    q_value1: jnp.ndarray
    q_value2: jnp.ndarray
    carry: tuple


class SACCriticNetwork(nn.Module):
    """A simple critic network used in SAC."""
    output_shape: int
    hidden_units: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        kernel_initializer = jax.nn.initializers.orthogonal(jnp.sqrt(2))
        bias_initializer = jax.nn.initializers.constant(0)

        state_q1 = forager_network(state, self.hidden_units)
        state_q2 = forager_network(state, self.hidden_units)

        q1 = nn.Dense(features=self.output_shape, kernel_init=kernel_initializer, bias_init=bias_initializer)(state_q1)
        q2 = nn.Dense(features=self.output_shape, kernel_init=kernel_initializer, bias_init=bias_initializer)(state_q2)
        return SacCriticOutput(q_value1=q1, q_value2=q2)

    def get_action_values(self, params, state: jnp.ndarray, action: jnp.ndarray) -> SacCriticOutput:
        """Calls the SAC critic network.

        This can be called using network_def.apply(..., method=network_def.critic).

        Args:
          state: An input state.

        Returns:
          A named tuple containing the Q-values for the state.
        """
        q1, q2 = self.apply(params, state)
        q1, q2 = jnp.squeeze(q1[action]), jnp.squeeze(q2[action])
        return SacCriticOutput(q_value1=q1, q_value2=q2)


class SACActorNetwork(nn.Module):
    """A simple critic network used in SAC."""
    output_shape: int
    hidden_units: int

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        kernel_initializer = jax.nn.initializers.orthogonal(jnp.sqrt(2))
        bias_initializer = jax.nn.initializers.constant(0)

        x = forager_network(state, self.hidden_units)
        actor_mean = nn.Dense(features=self.output_shape, kernel_init=kernel_initializer, bias_init=bias_initializer)(x)
        return actor_mean

    def sample(self, params, state: jnp.ndarray, key: jnp.ndarray) -> SacActorOutput:
        """Calls the SAC actor network.

        This can be called using network_def.apply(..., method=network_def.actor).

        Args:
          state: An input state.
          key: A PRNGKey to use to sample an action from the actor's output
            distribution.

        Returns:
          A named tuple containing a sampled action, the greedy action, and the
            likelihood of the sampled action.
        """

        actor_mean = self.apply(params, state)

        pi = tfp.Categorical(logits=actor_mean)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)

        return SacActorOutput(action, log_prob)
