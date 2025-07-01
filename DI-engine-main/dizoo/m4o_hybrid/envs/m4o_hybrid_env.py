import copy
import os
from typing import Dict, Optional

import gym
import m4o_hybrid
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict
from matplotlib import animation

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('m4o_hybrid')
class M4OHybridEnv(BaseEnv):
    default_env_id = ['M4O-v0']

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        env_id='M4O-v0',
        act_scale=True,
    )

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._act_scale = cfg.act_scale
        self._replay_path = None
        self._save_replay = False
        self._save_replay_count = 0
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1, ), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._eval_episode_return = 0
        obs = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: Dict) -> BaseEnvTimestep:
        if self._act_scale:
            action = [
                action['action_type'], [affine_transform(i, min_val=0, max_val=1) for i in action['action_args']]
            ]
        if self._save_replay:
            pass
            # self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)

        obs = to_ndarray(obs)
        if isinstance(obs, list):  # corner case
            for i in range(len(obs)):
                if len(obs[i].shape) == 0:
                    obs[i] = np.array([obs[i]])
            obs = np.concatenate(obs)
        assert isinstance(obs, np.ndarray) and obs.shape == (47, ) # state shape
        obs = obs.astype(np.float32)

        rew = to_ndarray([rew])  # wrapped to be transferred to a numpy array with shape (1,)
        if isinstance(rew, list):
            rew = rew[0]
        assert isinstance(rew, np.ndarray) and rew.shape == (1, )
        self._eval_episode_return += rew.item()

        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay:
                pass

        return BaseEnvTimestep(obs, rew, done, info)

    def random_action(self) -> Dict:
        raw_action = self._action_space.sample()
        return {'action_type': raw_action[0], 'action_args': raw_action[1]}


    def __repr__(self) -> str:
        return "DI-engine m4o hybrid Env"

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        pass

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        pass
