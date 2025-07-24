from PyExpUtils.collection.Collector import Collector

from environments.ForagerTemperature import ForagerTemperature as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem


class ForagerTemperature(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(self.seed, **self.env_params)
        self.actions = 4

        # get aperture size from env
        # should be initialized when Env() is called
        assert self.env.env._ap_size is not None
        ap_x, ap_y = self.env.env._ap_size

        # get number of object types from env
        num_objects = len(self.env.env._names)

        # if memory trace is used, we double the number of channels
        if self.env_params.get("memory_trace_lambda", 0.0) > 0:
            num_objects *= 2

        self.observations = (ap_x, ap_y, num_objects)
        self.gamma = 0.99
