import jax
import optax
import numpy as np
from algorithms.nn.components.RNNReplayBuffer import RNNReplayBuffer
import utils.chex as cxu

from abc import abstractmethod
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer

from algorithms.BaseAgent import BaseAgent
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities, sample


@cxu.dataclass
class AgentState:
    params: Any
    optim: optax.OptState


@checkpointable(("buffer", "steps", "state", "updates"))
class NNAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # ------------------------------
        # -- Configuration Parameters --
        # ------------------------------
        self.rep_params: Dict = params["representation"]
        self.optimizer_params: Dict = params["optimizer"]

        self.epsilon_steps = params.get("epsilon_steps")
        if self.epsilon_steps is not None:
            self.epsilon = params.get("epsilon")
            self.initial_epsilon = params.get("initial_epsilon", self.epsilon)
            self.final_epsilon = params.get("final_epsilon", self.epsilon)
            self.epsilon = self.initial_epsilon
        else:
            self.epsilon = params["epsilon"]
        assert self.epsilon is not None or (
            self.epsilon_steps is not None
            and self.initial_epsilon is not None
            and self.initial_epsilon is not None
            and self.final_epsilon is not None
        )
        self.reward_clip = params.get("reward_clip", 0)

        # ---------------------
        # -- NN Architecture --
        # ---------------------
        builder = NetworkBuilder(observations, self.rep_params, seed)
        self._build_heads(builder)
        if self.__class__.__name__ == "DRQN":
            self.phi = builder.getRecurrentFeatureFunction()
        else:
            self.phi = builder.getFeatureFunction()
        net_params = builder.getParams()
        self.net_params = net_params

        # ---------------
        # -- Optimizer --
        # ---------------
        self.optimizer = optax.adam(
            self.optimizer_params["alpha"],
            self.optimizer_params["beta1"],
            self.optimizer_params["beta2"],
            self.optimizer_params.get("eps", 1e-8),

        )
        opt_state = self.optimizer.init(net_params)

        # ------------------
        # -- Data ingress --
        # ------------------
        self.buffer_size = params["buffer_size"]
        self.batch_size = params["batch"]
        self.update_freq = params.get("update_freq", 1)
        self.minimum_replay_history = params.get("minimum_replay_history", self.batch_size)
        self.sequence_length = params.get("sequence_length", 1)

        self.normalizer_params = params.get("normalizer", {})

        self.buffer = (
            RNNReplayBuffer(self.buffer_size, self.n_step, self.rng, self.sequence_length)
            if params["buffer_type"] == "rnn_uniform"
            else build_buffer(
                buffer_type=params["buffer_type"],
                max_size=self.buffer_size,
                lag=self.n_step,
                rng=self.rng,
                config=params.get("buffer_config", {}),
            )
        )

        # --------------------------
        # -- Stateful information --
        # --------------------------
        self.state = AgentState(
            params=net_params,
            optim=opt_state,
        )

        self.steps = 0
        self.updates = 0

    # ------------------------
    # -- NN agent interface --
    # ------------------------

    @abstractmethod
    def _build_heads(self, builder: NetworkBuilder) -> None: ...

    @abstractmethod
    def _values(self, state: Any, x: np.ndarray) -> jax.Array: ...

    @abstractmethod
    def update(self) -> None: ...

    def policy(self, obs: np.ndarray) -> np.ndarray:
        q = self.values(obs)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    # --------------------------
    # -- Base agent interface --
    # --------------------------
    def values(self, x: np.ndarray, *args, **kwargs):
        x = np.asarray(x, dtype=np.float32)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q = self._values(self.state, x, *args, **kwargs)[0]

        else:
            q = self._values(self.state, x, *args, **kwargs)

        return jax.device_get(q)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray):  # type: ignore
        self.buffer.flush()
        x = np.asarray(x, dtype=np.float32)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.buffer.add_step(
            Timestep(
                x=x,
                a=a,
                r=None,
                gamma=self.gamma,
                terminal=False,
            )
        )
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

        self.buffer.add_step(
            Timestep(
                x=xp,
                a=a,
                r=r,
                gamma=self.gamma * gamma,
                terminal=False,
            )
        )

        if self.epsilon_steps is not None:
            self.epsilon = max(
                self.final_epsilon,
                self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * self.steps / self.epsilon_steps,
            )

        self.update()
        return a

    def end(self, r: float, extra: Dict[str, Any]):  # type: ignore
        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        self.buffer.add_step(
            Timestep(
                x=np.zeros(self.observations),
                a=-1,
                r=r,
                gamma=0,
                terminal=True,
            )
        )

        self.update()
