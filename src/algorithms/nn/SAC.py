from typing import Dict, Tuple, Any
import numpy as np
from algorithms.BaseAgent import BaseAgent
from PyExpUtils.collection.Collector import Collector
from algorithms.wrappers.sac import SACAgent


class SAC(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']
        self.seed = seed

        # set up the network parameters
        self.update_freq = params['update_freq']
        self.entropy_coeff = self.params['entropy_coeff']

        self.network_type = self.rep_params['type']
        self.hidden_size = self.rep_params['hidden']

        # set up the experience replay buffer
        self.buffer_type = params['buffer_type']
        self.buffer_size = params['buffer_size']
        self.buffer_config = params.get('buffer_config', {})

        self.batch_size = params['batch']
        self.min_replay_history = params.get('min_replay_history', 500)

        # set up the target network parameters
        self.tau = params['tau']
        self.target_update_period = params.get('target_update_period', 1)

        # optimizer parameters
        self.opt_type = self.optimizer_params['name'].lower()
        self.learning_rate = self.optimizer_params['alpha']
        self.beta1 = self.optimizer_params['beta1']
        self.beta2 = self.optimizer_params['beta2']
        self.alpha_learning_rate = None

        self.agent = SACAgent(
            action_shape=actions,
            observation_shape=observations,
            batch_size=self.batch_size,
            hidden_units=self.hidden_size,
            gamma=self.gamma,
            update_horizon=self.n_step,
            min_replay_history=self.min_replay_history,
            buffer_size=self.buffer_size,
            update_period=self.update_freq,
            tau=self.tau,
            target_update_period = self.target_update_period,
            learning_rate=self.learning_rate,
            alpha_learning_rate=self.alpha_learning_rate,
            seed=self.seed,
            entropy_coeff=self.entropy_coeff,
        )
        self.steps = 0

    def start(self, s: np.ndarray):
        return self.agent.begin_episode(s)

    def step(self, r: float, sp: np.ndarray | None, extra: Dict[str, Any]):
        self.steps += 1
        gamma = extra.get('gamma', 1.0)
        a = self.agent.step(r, sp, gamma)
        return a

    def __setstate__(self, state):
        self.__init__(*state['__args'])
        self.rng = state['rng']

        self.agent.buffer = state['buffer']
        self.steps = state['steps']
        self.agent.agent_state = state['agent_state']
        self.agent.updates = state['updates']

    def __getstate__(self):
        return {
            '__args': (self.observations, self.actions, self.params, self.collector, self.seed),
            'rng': self.agent._rng,
            'buffer': self.agent.buffer,
            'steps': self.steps,
            'agent_state': self.agent.agent_state,
            'updates': self.agent.updates,
        }
    