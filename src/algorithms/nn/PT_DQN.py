import operator
from functools import partial
from typing import Any, Dict, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer
from ReplayTables.ReplayBuffer import Batch

import utils.chex as cxu
from algorithms.nn.NNAgent import NNAgent
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.jax import huber_loss
from utils.policies import sample


@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState
    optim_p: optax.OptState


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        "delta": delta,
    }


@checkpointable(("buffer_p",))
class PT_DQN(NNAgent):
    def __init__(
        self,
        observations: Tuple,
        actions: int,
        params: Dict,
        collector: Collector,
        seed: int,
    ):
        super().__init__(observations, actions, params, collector, seed)
        # set up the target network parameters
        self.target_refresh = params["target_refresh"]
        self.tau = params.get("tau", 1.0)
        self.w0_regularization = params.get("w0_regularization", 0.0)

        self.initial_params = self.state.params

        # ---------------
        # -- Optimizer --
        # ---------------
        self.optimizer_p = optax.sgd(
            self.optimizer_params["alpha_p"],
        )
        self.optimizer = optax.adam(
            self.optimizer_params["alpha"],
            self.optimizer_params["beta1"],
            self.optimizer_params["beta2"],
            self.optimizer_params.get("eps", 1e-8),

        )
        p_net_params = {
            'phi': {
                k: v for k, v in self.net_params["phi"].items() if "permanent" in k
            },
            'q_p': self.net_params['q_p'],
        }
        t_net_params = {
            'phi': {
                k: v for k, v in self.net_params["phi"].items() if "transient" in k
            },
            'q': self.net_params['q'],
        }
        self.state = AgentState(
            params=self.state.params,
            target_params=self.state.params,
            optim=self.optimizer.init(t_net_params),
            optim_p=self.optimizer_p.init(p_net_params),
        )

        self.buffer_p_size = params["buffer_p_size"]

        self.buffer_p = build_buffer(
            buffer_type=params["buffer_type"],
            max_size=self.buffer_p_size,
            lag=self.n_step,
            rng=self.rng,
            config=params.get("buffer_config", {}),
        )

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray):  # type: ignore
        self.buffer.flush()
        self.buffer_p.flush()
        x = np.asarray(x, dtype=np.float32)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        t = Timestep(
            x=x,
            a=a,
            r=None,
            gamma=self.gamma,
            terminal=False,
        )
        self.buffer.add_step(t)
        self.buffer_p.add_step(t)
        return a

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]):  # type: ignore
        a = -1

        # sample next action
        if xp is not None:
            xp = np.asarray(xp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get("gamma", 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        t = Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
        )
        self.buffer.add_step(t)
        self.buffer_p.add_step(t)

        if self.epsilon_steps is not None:
            self.epsilon = max(
                self.final_epsilon,
                self.initial_epsilon
                - (self.initial_epsilon - self.final_epsilon)
                * self.steps
                / self.epsilon_steps,
            )

        self.update()
        return a

    def end(self, r: float, extra: Dict[str, Any]):  # type: ignore
        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)
        t = Timestep(
            x=np.zeros(self.observations),
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
        )
        self.buffer.add_step(t)
        self.buffer_p.add_step(t)

        self.update()

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        self.q_p = builder.addHead(lambda: hk.Linear(self.actions, name="q_p"))
        self.q = builder.addHead(lambda: hk.Linear(self.actions, name="q"))

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        phi_p, phi_t = self.phi(state.params, x).out
        return self.q_p(state.params, phi_p) + self.q(state.params, phi_t)

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.minimum_replay_history:
            return

        self.updates += 1

        batch = self.buffer.sample(self.batch_size)
        weights = self.buffer.isr_weights(batch.trans_id)
        self.state, metrics = self._computeUpdate(
            self.state, batch, weights, self.initial_params
        )

        metrics = jax.device_get(metrics)

        priorities = metrics["delta"]
        self.buffer.update_batch(batch, priorities=priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

        if self.updates % self.target_refresh == 0:
            self._sync_params()

        # TODO: add permanent network update

    def _sync_params(self):
        def _polyak_weights(target_p, online_p):
            return self.tau * online_p + (1 - self.tau) * target_p

        self.state.target_params = jax.tree_map(
            _polyak_weights,
            self.state.target_params,
            self.state.params,
        )

    # -------------
    # -- Updates --
    # -------------
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(
        self,
        state: AgentState,
        batch: Batch,
        weights: jax.Array,
        initial_params: jax.Array,
    ):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(
            state.params, state.target_params, initial_params, batch, weights
        )

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
            optim_p=state.optim_p,
        )

        return new_state, metrics

    def _loss(
        self,
        params: hk.Params,
        target: hk.Params,
        initial_params: hk.Params,
        batch: Batch,
        weights: jax.Array,
    ):
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(target, batch.xp).out

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        loss += self.w0_regularization * jax.tree.reduce(
            operator.add,
            jax.tree.map(
                lambda p, ip: jnp.sum(jnp.square(p - ip)),
                params,
                initial_params,
            ),
        )

        return loss, metrics
