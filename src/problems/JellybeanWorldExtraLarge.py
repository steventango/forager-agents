from PyExpUtils.collection.Collector import Collector

from environments.JellybeanWorldExtraLarge import JellybeanWorldExtraLarge as Env
from environments.JellybeanWorldExtraLarge import items
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class JellybeanWorldExtraLarge(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(self.seed, **self.env_params)
        self.actions = 4

        ap = self.env.config.vision_range * 2 + 1

        self.observations = (ap, ap, len(items))
        self.gamma = 0.99
