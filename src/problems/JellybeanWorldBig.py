from PyExpUtils.collection.Collector import Collector

from environments.JellybeanWorldBig import JellybeanWorldBig as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class JellyBeanWorldBig(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(self.seed, **self.env_params)
        self.actions = 4

        ap = self.env.config.vision_range * 2 + 1

        self.observations = (ap, ap, len(self.env.config.items))
        self.gamma = 0.9
