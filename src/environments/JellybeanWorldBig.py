from math import log, pi
from typing import Any, Dict, Tuple

import jbw
from RlGlue import BaseEnvironment


items = []
items.append(
    jbw.Item(
        "JellyBean",
        [1.64, 0.54, 0.40],
        [0.82, 0.27, 0.20],
        [0, 0],
        [0, 0],
        False,
        0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT,
        intensity_fn_args=[log(0.1)],
        interaction_fns=[
            [jbw.InteractionFunction.ZERO],
            [jbw.InteractionFunction.ZERO],
        ],
    )
)
items.append(
    jbw.Item(
        "Onion",
        [0.68, 0.01, 0.99],
        [0.68, 0.01, 0.99],
        [0, 0],
        [0, 0],
        False,
        0.0,
        intensity_fn=jbw.IntensityFunction.CONSTANT,
        intensity_fn_args=[log(0.1)],
        interaction_fns=[
            [jbw.InteractionFunction.ZERO],
            [jbw.InteractionFunction.ZERO],
        ],
    )
)


class JellybeanWorldBig(BaseEnvironment):
    def __init__(self, seed):
        self.config = jbw.SimulatorConfig(
            max_steps_per_movement=1,
            vision_range=5,
            allowed_movement_directions=[
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
                jbw.ActionPolicy.ALLOWED,
            ],
            allowed_turn_directions=[
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
                jbw.ActionPolicy.DISALLOWED,
            ],
            no_op_allowed=False,
            patch_size=32,
            mcmc_num_iter=4000,
            items=items,
            agent_color=[0.0, 0.0, 1.0],
            collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
            agent_field_of_view=2 * pi,
            decay_param=0,
            diffusion_param=0,
            deleted_item_lifetime=2000,
            seed=seed,
        )
        sim = jbw.Simulator(sim_config=self.config)
        self.agent = jbw.Agent(sim, None)
        self.previous_items = None

    def start(self) -> Any:
        self.previous_items = self.agent.collected_items()
        return self.agent.vision()

    def step(self, action: int) -> Tuple[float, Any, bool, Dict[str, Any]]:
        self.agent.move(action)
        items = self.agent.collected_items()
        jellybean_delta = items[0] - self.previous_items[0]
        onion_delta = items[1] - self.previous_items[1]
        self.previous_items = items
        reward = jellybean_delta - onion_delta
        obs = self.agent.vision()
        return reward, obs, False, {}
