from copy import deepcopy
from functools import partial
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.ReplayBuffer import Batch, LaggedTimestep
from jax.tree_util import tree_flatten
from algorithms.nn.NNAgent import NNAgent
from algorithms.nn.components.RNNReplayBuffer import CarryBatch
from representations.networks import NetworkBuilder
from utils.jax import huber_loss, mse_loss
from ReplayTables.interface import Timestep, TransIds
from utils.policies import egreedy_probabilities, sample

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp
import utils.chex as cxu

@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    #return huber_loss(1.0, q[a], target), {
    return mse_loss(q[a], target), {
        'delta': delta,
    }

class DRQN(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = int(params['target_refresh'])
        self.train_use_all_steps = params.get('train_use_all_steps', True)
        self.burn_in_steps = int(params.get('burn_in_steps', 0))
        self.trainable_steps = self.sequence_length - self.burn_in_steps
        if self.trainable_steps < 1:
            raise Exception("Sequence length must be longer than burn in steps")
        self.carry = None
        self.state = AgentState(
            params=self.state.params,
            target_params=deepcopy(self.state.params), # without deepcopy, load_from_checkpoint overwrites params with target_params
            optim=self.state.optim,
        )

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        self.q = builder.addHead(lambda: hk.Linear(self.actions, name='q'))

    def values(self, x: np.ndarray, carry=None):
        x = np.asarray(x)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q, carry, initial_carry = self._values(self.state, x, carry=carry)
            q = q[0]

        else:
            q, carry, initial_carry = self._values(self.state, x, carry=carry)

        return jax.device_get(q), jax.device_get(carry), jax.device_get(initial_carry)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        q, self.carry, _ = self.values(obs, carry=self.carry)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    def ext_policy(self, obs: np.ndarray, carry: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, carry, _ = self.values(obs, carry=carry)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi, q, carry

    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array, carry: jax.Array = None): # type: ignore
        phi = self.phi(state.params, x, carry=carry)
        return self.q(state.params, phi[0][:, -1]), phi[1][:, -1], phi[2]

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size * self.sequence_length:
            return

        self.updates += 1

        batch = self.buffer.sample_sequences(self.batch_size)
        weights_shape = batch.trans_id.shape
        weights = self.buffer.isr_weights(TransIds(np.array(batch.trans_id).ravel())).reshape(weights_shape)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        self.buffer.update_batch(batch, priorities=priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params   # deepcopy not needed here because optax.apply_updates produces a new pytree as params at each update, so params becomes unlinked from target_params

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: LaggedTimestep, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
        )

        return new_state, metrics

    # Loss is computed for the final action in the sequence
    def _loss(self, params: hk.Params, target: hk.Params, batch: CarryBatch, weights: jax.Array):
        x = batch.x
        xp = batch.xp
        carry = batch.carry
        carryp = batch.carryp
        term = batch.terminal
        reset = batch.reset
        a = batch.a
        r = batch.r
        gamma = batch.gamma

        # Perform burn-in
        if self.burn_in_steps > 0:
            b_x, x = jnp.hsplit(x, [self.burn_in_steps])
            b_xp, xp = jnp.hsplit(xp, [self.burn_in_steps])
            _, term = jnp.hsplit(term, [self.burn_in_steps])
            b_reset, reset = jnp.hsplit(reset, [self.burn_in_steps])
            b_carry, carry = jnp.hsplit(carry, [self.burn_in_steps])
            b_carryp, carryp = jnp.hsplit(carryp, [self.burn_in_steps])
            _, a = jnp.hsplit(a, [self.burn_in_steps])
            _, r = jnp.hsplit(r, [self.burn_in_steps])
            _, gamma = jnp.hsplit(gamma, [self.burn_in_steps])
            _, weights = jnp.hsplit(weights, [self.burn_in_steps])

            carry = carry.at[:, 0].set(jax.lax.stop_gradient(self.phi(params, b_x, carry=b_carry, reset=b_reset, is_target=False)[1][:, -1, ...]))
            carryp = carryp.at[:, 0].set(jax.lax.stop_gradient(self.phi(target, b_xp, carry=b_carryp, reset=b_reset, is_target=True)[1][:, -1, ...]))

        phi = self.phi(params, x, carry=carry, reset=reset, is_target=False)[0]
        phi_p = self.phi(target, xp, carry=carryp, reset=reset, is_target=True)[0]

        if self.rep_params.get("frozen"):
            phi = jax.lax.stop_gradient(phi)

        if self.train_use_all_steps:
            # After the representation layer, we use all
            qs = self.q(params, phi)
            qsp = self.q(target, phi_p)

            qs = qs.reshape(-1, qs.shape[-1])
            qsp = qsp.reshape(-1, qsp.shape[-1])

            weights = weights.ravel()
            a = a.ravel()
            r = r.ravel()
            gamma = gamma.ravel()

        else:
            weights = weights[:, -1]

            # After the representation layer, we just use the last
            qs = self.q(params, phi[:, -1, ...])
            qsp = self.q(target, phi_p[:, -1, ...])

            a = a[:, -1, ...]
            r = r[:, -1, ...]
            gamma = gamma[:, -1, ...]

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, a, r, gamma, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray): # type: ignore
        self.is_successful = False

        self.carry = None
        self.buffer.flush()
        x = np.asarray(x)
        x = self.normalize_state(x)
        carry = self.values(x)[2]
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.buffer.add_step(Timestep(
            x=x,
            a=a,
            r=None,
            gamma=self.gamma,
            terminal=False,
            extra={
                'carry': carry,
                'carryp': self.carry,
                'reset': True
                }
        ))

        return a

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]): # type: ignore
        a = -1
        carry = self.carry

        # sample next action
        if xp is not None:
            xp = np.asarray(xp)
            xp = self.normalize_state(xp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        self.buffer.add_step(Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
            extra={
                'carry': carry,
                'carryp': self.carry,
                'reset': False
                }
        ))

        self.update()
        return a

    def end(self, r: float, extra: Dict[str, Any]): # type: ignore
        self.is_successful = extra.get('success', False)

        carry = self.carry
        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        self.buffer.add_step(Timestep(
            x=np.zeros(self.observations),
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
            extra={
                'carry': carry,
                'carryp': self.carry,
                'reset': False
                }
        ))

        self.update()
