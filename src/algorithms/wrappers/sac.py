import functools
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer

import utils.chex as cxu
from utils.jax import mse_loss
from algorithms.wrappers.sac_network import SACActorNetwork, SACCriticNetwork

@cxu.dataclass
class AgentState:
    critic_params: Any
    target_params: Any
    policy_params: Any
    log_alpha: Any
    critic_optim: optax.OptState
    policy_optim: optax.OptState
    alpha_optim: optax.OptState

@functools.partial(jax.jit, static_argnums=(0, 1, 3, 4, 5))
def train(critic: nn.Module,
          actor: nn.Module,
          log_alpha: jnp.ndarray,
          critic_optimizer: optax.GradientTransformation,
          actor_optimizer: optax.GradientTransformation,
          alpha_optimizer: optax.GradientTransformation,
          critic_params: jnp.ndarray,
          policy_params: jnp.ndarray,
          target_params: jnp.ndarray,
          critic_optim: optax.OptState,
          policy_optim: optax.OptState,
          alpha_optim: optax.OptState,
          key: jnp.ndarray,
          states: jnp.ndarray,
          actions: jnp.ndarray,
          next_states: jnp.ndarray,
          rewards: jnp.ndarray,
          terminals: jnp.ndarray,
          gamma: float,
          target_entropy: float) -> Mapping[str, Any]:

    def critic_loss_fn(critic_params: jnp.ndarray, policy_params: jnp.ndarray, target_params: jnp.ndarray,
                       state: jnp.ndarray, action: jnp.ndarray, reward: jnp.ndarray, next_state: jnp.ndarray,
                       terminal: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:

        # J_Q(\theta) from equation (2, 3) of https://arxiv.org/pdf/2112.02852.pdf.
        q1, q2 = critic.get_action_values(critic_params, state, action)

        actor_mean = actor.apply(policy_params, next_state)
        probs = jax.nn.softmax(actor_mean)
        log_probs = jax.nn.log_softmax(actor_mean)

        alpha_value = jnp.exp(log_alpha)

        target_q1, target_q2 = critic.apply(target_params, next_state)
        target_q1 = probs @ (target_q1 - alpha_value * log_probs)
        target_q2 = probs @ (target_q2 - alpha_value * log_probs)
        target_q = jnp.minimum(target_q1, target_q2)

        target = reward + (1. - terminal) * gamma * target_q
        target = jax.lax.stop_gradient(target)

        critic_loss_1 = mse_loss(q1, target)
        critic_loss_2 = mse_loss(q2, target)

        critic_loss = 0.5 * (critic_loss_1 + critic_loss_2)
        return critic_loss

    def policy_loss_fn(
            policy_params, critic_params, log_alpha: jnp.ndarray,
            state: jnp.ndarray) -> jnp.ndarray:

        # J_{\pi}(\phi) from equation (5) in paper.
        actor_mean = actor.apply(policy_params, state)
        probs = jax.nn.softmax(actor_mean)
        log_probs = jax.nn.log_softmax(actor_mean)

        alpha_value = jnp.exp(log_alpha)
        q1, q2 = critic.apply(critic_params, state)

        q1 = alpha_value * log_probs - q1
        q2 = alpha_value * log_probs - q2

        q1 = jax.lax.stop_gradient(q1)
        q2 = jax.lax.stop_gradient(q2)

        # make a vector of all values
        policy_loss = jnp.minimum(probs @ q1, probs @ q2)
        return policy_loss

    def alpha_loss_fn(log_alpha: jnp.ndarray, policy_params: jnp.ndarray,
                      state: jnp.ndarray) -> jnp.ndarray:

        # J(\alpha) from equation (7) in paper.
        actor_mean = actor.apply(policy_params, state)
        probs = jax.nn.softmax(actor_mean)
        log_probs = jax.nn.log_softmax(actor_mean)

        entropy_diff = -log_probs - target_entropy
        entropy_diff = probs @ entropy_diff
        entropy_diff = jax.lax.stop_gradient(entropy_diff)

        alpha_loss = jnp.mean(log_alpha * entropy_diff)
        return alpha_loss

    batch_size = states.shape[0]
    rng = jnp.stack(jax.random.split(key, num=batch_size))

    # updating critic
    critic_grad_fn = jax.vmap(jax.grad(critic_loss_fn, has_aux=False),
                              in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
    critic_grad = critic_grad_fn(critic_params, policy_params, target_params,
                                 states, actions, rewards, next_states, terminals, rng)
    critic_grad = jax.tree_map(functools.partial(jnp.mean, axis=0), critic_grad)

    # updating policy
    actor_grad_fn = jax.vmap(jax.grad(policy_loss_fn, has_aux=False),
                             in_axes=(None, None, None, 0))
    actor_grad = actor_grad_fn(policy_params, critic_params, log_alpha, states)
    actor_grad = jax.tree_map(functools.partial(jnp.mean, axis=0), actor_grad)

    # adjust temperature alpha
    if alpha_optim is not None:
        alpha_grad_fn = jax.vmap(jax.grad(alpha_loss_fn, has_aux=False),
                                 in_axes=(None, None, 0))
        alpha_grad = alpha_grad_fn(log_alpha, policy_params, states)
        alpha_grad = jax.tree_map(functools.partial(jnp.mean, axis=0), alpha_grad)[0]

    # update the parameters
    updates, critic_optim = critic_optimizer.update(critic_grad, critic_optim, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)

    updates, policy_optim = actor_optimizer.update(actor_grad, policy_optim, policy_params)
    policy_params = optax.apply_updates(policy_params, updates)

    if alpha_optim is not None:
        updates, alpha_optim = alpha_optimizer.update(alpha_grad, alpha_optim, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, updates)

    return AgentState(
        critic_params=critic_params,
        critic_optim=critic_optim,
        target_params=target_params,
        policy_params=policy_params,
        policy_optim=policy_optim,
        log_alpha=log_alpha,
        alpha_optim=alpha_optim
    )

@functools.partial(jax.jit, static_argnums=0)
def select_action(policy, params, state, rng):
    rng, rng2 = jax.random.split(rng)
    action, _ = policy.sample(params, state, rng2)
    return rng, action


class SACAgent:
    """A JAX implementation of the SAC agent."""

    def __init__(self,
                 action_shape,
                 observation_shape,
                 batch_size=32,
                 buffer_size=1000000,
                 buffer_type='uniform',
                 buffer_config={},
                 hidden_units=256,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=500,
                 update_period=1,
                 tau=0.005,
                 target_update_period=100,
                 target_entropy=None,
                 learning_rate=0.0003,
                 alpha_learning_rate=0.0003,
                 beta1=0.9,
                 beta2=0.999,
                 seed=None,
                 entropy_coeff=None):
        r"""Initializes the agent and constructs the necessary components.

        Args:
          action_shape: int or tuple, dimensionality of the action space.
          observation_shape: tuple of ints describing the observation shape.
          batch_size: int, number of samples each update.
          buffer_size: int, the size of the replay buffer.
          network_type: TwoLayerRelu or Minatar.
          hidden_units: int, number of hidden units in the network.
          gamma: float, discount factor with the usual RL meaning.
          update_horizon: int, horizon at which updates are performed, the 'n' in
            n-step update.
          min_replay_history: int, number of transitions that should be experienced
            before the agent begins training its value function.
          update_period: int, period between DQN updates.
          tau: float, smoothing coefficient for target
            network updates (\tau in paper) when in 'soft' mode.
          target_entropy: float or None, the target entropy for training alpha. If
            None, it will default to the half the negative of the number of action
            dimensions.
          optimizer: str, name of optimizer to use.
          seed: int, a seed for SAC's internal RNG, used for initialization and
            sampling actions.
        """
        if target_entropy is None:
            target_entropy = -action_shape

        self.action_shape = action_shape
        self.observation_shape = tuple(observation_shape)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.n_step = update_horizon
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.tau = tau
        self.target_update_period = target_update_period
        self.target_entropy = target_entropy
        self.learning_rate = learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.hidden_units = hidden_units
        self.entropy_coeff = entropy_coeff

        self.training_steps = 0
        self.updates = 0

        self._rng = jax.random.PRNGKey(seed)
        state_shape = self.observation_shape
        self.state = np.zeros(state_shape)
        self.buffer = build_buffer(
            buffer_type=buffer_type,
            max_size=self.buffer_size,
            lag=self.n_step,
            rng=np.random.default_rng(seed),
            config=buffer_config,
        )

        self._build_networks_and_optimizer()

    def _build_networks_and_optimizer(self):
        self._rng, init_key = jax.random.split(self._rng)

        self.critic = SACCriticNetwork(self.action_shape, self.hidden_units)
        self.actor = SACActorNetwork(self.action_shape, self.hidden_units)

        critic_params = self.critic.init(init_key, self.state)
        self.critic_optimizer = optax.adam(self.learning_rate)
        critic_opt = self.critic_optimizer.init(critic_params)

        actor_params = self.actor.init(init_key, self.state)
        self.actor_optimizer = optax.adam(self.learning_rate)
        actor_opt = self.actor_optimizer.init(actor_params)

        # alpha network
        if self.entropy_coeff is None:
            log_alpha = jnp.zeros(1)
            self.alpha_optimizer = optax.adam(self.alpha_learning_rate)
            alpha_opt = self.alpha_optimizer.init(log_alpha)
        else:
            alpha_opt = None
            self.alpha_optimizer = None
            log_alpha = jnp.log(self.entropy_coeff)

        self.agent_state = AgentState(
            critic_params=critic_params,
            target_params=critic_params,
            policy_params=actor_params,
            log_alpha=log_alpha,
            critic_optim=critic_opt,
            policy_optim=actor_opt,
            alpha_optim=alpha_opt)


    def _maybe_sync_weights(self):
        """Syncs the target weights with the online weights."""
        target_params = optax.incremental_update(self.agent_state.critic_params,
                                        self.agent_state.target_params, self.tau)

        self.agent_state.target_params = target_params

    def begin_episode(self, observation):
        """Returns the agent's first action for this episode.

        Args:
          observation: numpy array, the environment's initial observation.

        Returns:
          np.ndarray, the selected action.
        """
        self.buffer.flush()
        self._train_step()

        self._rng, self.action = select_action(self.actor, self.agent_state.policy_params, observation, self._rng)
        self.buffer.add_step(Timestep(
            x=observation,
            a=self.action,
            r=None,
            gamma=self.gamma,
            terminal=False,
        ))

        self.action = np.asarray(self.action)
        return self.action

    def step(self, reward, observation, gamma):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
          reward: float, the reward received from the agent's most recent action.
          observation: numpy array, the most recent observation.

        Returns:
          int, the selected action.
        """
        self._train_step()

        self._rng, self.action = select_action(self.actor, self.agent_state.policy_params, observation, self._rng)
        self.buffer.add_step(Timestep(
            x=observation,
            a=self.action,
            r=reward,
            gamma=self.gamma * gamma,
            terminal=False,
        ))
        self.action = np.asarray(self.action)
        return self.action


    def _train_step(self):
        """Runs a single training step.

        Runs training if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online_network to target_network if training steps
        is a multiple of target update period.
        """
        if self.buffer.size() > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                batch = self.buffer.sample(self.batch_size)
                self._rng, key = jax.random.split(self._rng)

                self.agent_state = train(
                    self.critic, self.actor, self.agent_state.log_alpha,
                    self.critic_optimizer, self.actor_optimizer, self.alpha_optimizer,
                    self.agent_state.critic_params, self.agent_state.policy_params,
                    self.agent_state.target_params, self.agent_state.critic_optim,
                    self.agent_state.policy_optim, self.agent_state.alpha_optim,
                    key, batch.x, batch.a, batch.xp, batch.r, batch.terminal,
                    self.gamma, self.target_entropy)

                self.updates += 1
                self._maybe_sync_weights()

        self.training_steps += 1


    def __getstate__(self) -> object:
        return

    