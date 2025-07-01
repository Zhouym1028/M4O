import numpy as np
import pytest
from dizoo.m4o_hybrid.envs import M4OHybridEnv
from easydict import EasyDict

@pytest.mark.envtest
class TestGymHybridEnv:

    def test_naive(self):
        env = M4OHybridEnv(
            EasyDict(
                {
                    'env_id': 'M4O-v0',
                    'act_scale': False,
                    'save_replay_gif': False,
                    'replay_path_gif': None,
                    'replay_path': None
                }
            )
        )
        # env.enable_save_replay('./video')
        env.seed(314, dynamic_seed=False)
        assert env._seed == 314
        obs = env.reset()
        assert obs.shape == (env._env.num_V2V + env._env.num_V2I - 2, )
        for i in range(200):
            random_action = env.random_action()
            print('random_action', random_action)
            timestep = env.step(random_action)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (env._env.num_V2V + env._env.num_V2I - 2, )
            assert timestep.reward.shape == (1, )
            # assert timestep.info['action_args_mask'].shape == (3, 2)
            if timestep.done:
                print('reset env')
                env.reset()
        print(env.observation_space, env.action_space, env.reward_space)
        env.close()
