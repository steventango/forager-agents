from typing import Any

import numpy as np
from forager.config import ForagerConfig
from forager.Env import ForagerEnv
from forager.interface import Coords
from forager.objects import ForagerObject
from RlGlue import BaseEnvironment


class Jellybean(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name="jellybean")

        self.blocking = False
        self.collectable = True
        self.target_location = loc
        self.color = np.array((162, 53, 40), dtype=np.uint8)

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 1


class Onion(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name="onion")

        self.blocking = False
        self.collectable = True
        self.target_location = loc
        self.color = np.array((103, 2, 150), dtype=np.uint8)

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return -1


class ForagerExtraLarge(BaseEnvironment):
    def __init__(self, seed: int, aperture: int):
        config = ForagerConfig(
            size=(1000, 1000),
            object_types={
                "jellybean": Jellybean,
                "onion": Onion,
            },
            aperture=aperture,
            seed=seed,
        )
        self.env = ForagerEnv(config)
        self.env.generate_objects(0.1, "jellybean")
        self.env.generate_objects(0.1, "onion")

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (r, obs.astype(np.float32), False, {})

    def render(self):
        return self.env.render()
