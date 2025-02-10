import numpy as np
from typing import Any
from RlGlue import BaseEnvironment
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Morel, Oyster

class Forager(BaseEnvironment):
    def __init__(self, seed: int, aperture: int):
        config = ForagerConfig(
            size=(16, 8),
            object_types={
                "morel": Morel,
                "oyster": Oyster,
            },
            aperture=aperture,
            seed=seed
        )
        self.env = ForagerEnv(config)

        # because sample_unpopulated only does 10 tries, sometimes collisions will happen
        # so we need to set freq > 1 to ensure that we get the expected number of objects
        self.env.generate_objects(6.0, "morel", (2, 2), (6, 6))
        self.env.generate_objects(6.0, "oyster", (10, 2), (14, 6))

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (r, obs.astype(np.float32), False, {})

    def render(self):
        return self.env.render()
