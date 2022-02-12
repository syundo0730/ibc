from typing import Tuple, Optional, Dict
from math import pi

import gym
from gym import spaces
from gym.envs import registration
import numpy as np

_TOW_PI = 2 * pi


class RobotArmEnv(gym.Env):
    _DEFAULT_RESET_POSITION = np.array((0.2, 0))

    def __init__(self):
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # dx, dy
        # image_size = (240, 320)
        image_size = None
        self.observation_space = self._create_observation_space(image_size)
        self._dummy_observation = self._create_dummy_observation(image_size)

    @staticmethod
    def _create_observation_space(image_size: Optional[Tuple[float, float]]) -> spaces.Dict:
        obs_scape_dict = {
            'joint_angles': spaces.Box(
                low=-_TOW_PI, high=_TOW_PI, shape=(2,)
            ),
            'end_effector_pos': spaces.Box(
                low=-.3, high=.3, shape=(2,)
            ),
        }
        if image_size is not None:
            obs_scape_dict['rgb'] = spaces.Box(
                low=0, high=255,
                shape=(*image_size, 3),
                dtype=np.uint8)
        return spaces.Dict(obs_scape_dict)

    @staticmethod
    def _create_dummy_observation(image_size: Optional[Tuple[float, float]]) -> Dict:
        obs = {
            'joint_angles': np.zeros(2),
            'end_effector_pos': np.zeros(2),
        }
        if image_size:
            obs['rgb'] = np.zeros((*image_size, 3)),
        return obs

    def get_metrics(self, num_episodes):
        return [], None

    def step(self, action):
        return self._dummy_observation, 0, False, {}

    def reset(self):
        return self._dummy_observation

    def render(self, mode='human'):
        pass


registration.register(
    id='ScalaArm-v0',
    entry_point=RobotArmEnv,
)
