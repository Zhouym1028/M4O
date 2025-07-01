import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import env
from stable_baselines3 import A2C,DQN,PPO
np.random.seed(1)


# data for state
one_data = []
for _ in range(1):
    # computational intensity (1000)
    data_bit = [(np.random.uniform(2.4e9, 4.8e9)) for _ in range(10)]

    # change intensity here
    # data_bit = [x * 3 for x in data_bit]

    data_cpu = [(np.random.uniform(1.2e9, 1.6e9)) for _ in range(7)]

    # positions and velocities of vehicles
    vehicle_coords = []
    for _ in range(10):
        x = np.random.uniform(-200, 200)
        y = np.random.uniform(-10, 10)
        # velocity of vehicles
        z = np.random.uniform(-20, 20)

        vehicle_coords.extend([x, y, z])

    large_array_ = data_bit + data_cpu + vehicle_coords
    one_data.append(large_array_)

one_data = np.array(one_data)

# create environment
env = env.M4OEnv(one_data)

model = A2C('MlpPolicy',
            env,
            learning_rate=5e-4,
            # the path to store the training result
            tensorboard_log='a2c_1',
            verbose=1)

# training
model.learn(total_timesteps=100000)


