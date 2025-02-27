from glob import glob
from typing import Any

import numpy as np
import pandas as pd
from forager.config import ForagerConfig, ObjectFactory
from forager.Env import ForagerEnv
from forager.interface import Coords
from forager.objects import ForagerObject
from RlGlue import BaseEnvironment

DATA_PATH = "data/ECA_non-blended_custom"
FILE_PATHS = sorted(glob(f"{DATA_PATH}/TG_*.txt"))


def load_data(file_path: str):
    df = pd.read_csv(file_path, skiprows=21)
    df.columns = df.columns.str.strip()
    df = df[df["Q_TG"] == 0]
    df["mean_temperature"] = df["TG"] / 10
    df["normalized_mean_temperature"] = (df["mean_temperature"] - df["mean_temperature"].min()) / (
        df["mean_temperature"].max() - df["mean_temperature"].min()
    ) * 2 - 1
    df["date"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
    return df["normalized_mean_temperature"].to_numpy()


def hot_factory(rewards: np.ndarray, repeat: int) -> ObjectFactory:
    class Hot(ForagerObject):
        def __init__(self, loc: Coords | None = None):
            super().__init__(name="hot")

            self.blocking = False
            self.collectable = True
            self.target_location = loc
            self.color = np.array((255, 0, 255), dtype=np.uint8)
            self.rewards = rewards
            self.repeat = repeat

        def regen_delay(self, rng: np.random.Generator, clock: int) -> int | None:
            self.target_location = self.current_location
            return int(max(0, rng.normal(10, 1)))

        def reward(self, rng: np.random.Generator, clock: int) -> float:
            return self.rewards[clock // self.repeat % len(self.rewards)]

    return Hot


def cold_factory(rewards: np.ndarray, repeat: int) -> ObjectFactory:
    class Cold(ForagerObject):
        def __init__(self, loc: Coords | None = None):
            super().__init__(name="cold")

            self.blocking = False
            self.collectable = True
            self.target_location = loc
            self.color = np.array((0, 255, 255), dtype=np.uint8)
            self.rewards = rewards
            self.repeat = repeat

        def regen_delay(self, rng: np.random.Generator, clock: int) -> int | None:
            self.target_location = self.current_location
            return int(max(0, rng.normal(10, 1)))

        def reward(self, rng: np.random.Generator, clock: int) -> float:
            return self.rewards[clock // self.repeat % len(self.rewards)]

    return Cold


class ForagerTemperature(BaseEnvironment):
    def __init__(self, seed: int, aperture: int):
        assert 0 <= seed < len(FILE_PATHS)
        rewards = load_data(FILE_PATHS[seed])
        repeat = 100
        config = ForagerConfig(
            size=(15, 15),
            object_types={
                "hot": hot_factory(rewards, repeat),
                "cold": cold_factory(-rewards, repeat),
            },
            aperture=aperture,
            seed=seed,
        )
        self.env = ForagerEnv(config)
        # 012345678901234
        # ___AA_____BB___
        self.env.generate_objects(0.5, "hot", (0, 3), (15, 5))
        self.env.generate_objects(0.5, "cold", (0, 10), (15, 12))

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (r, obs.astype(np.float32), False, {})

    def render(self):
        return self.env.render(mode="world")
