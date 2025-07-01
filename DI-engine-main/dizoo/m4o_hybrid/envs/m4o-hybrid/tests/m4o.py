import time
import gym
import m4o_hybrid
from m4o_hybrid.environments import M4OEnv
import numpy as np
# np.random.seed(0)

if __name__ == '__main__':
    env = gym.make('M4O-v0')

    env.reset()

