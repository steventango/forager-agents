import numpy as np
from typing import Any
from RlGlue import BaseEnvironment
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Truffle, Oyster

class Forager(BaseEnvironment):
    def __init__(self, seed: int, aperture: int):
        config = ForagerConfig(
            size=(16, 8),
            object_types={
                "truffle": Truffle,
                "oyster": Oyster,
            },
            aperture=aperture,
            seed=seed
        )
        self.env = ForagerEnv(config)
        size = config.size
        truffle_locations = np.zeros(size)
        truffle_locations[2:6, 2:6] = 1
        truffle_locations = np.ravel_multi_index(np.where(truffle_locations), size, order="F")

        self.env.generate_objects_locations(2.0, "truffle", truffle_locations)

        oyster_locations = np.zeros(size)
        oyster_locations[10:14, 2:6] = 1
        oyster_locations = np.ravel_multi_index(np.where(oyster_locations), size, order="F")
        self.env.generate_objects_locations(2.0, "oyster", oyster_locations)

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (r, obs.astype(np.float32), False, {})

    def render(self):
        return self.env.render()
