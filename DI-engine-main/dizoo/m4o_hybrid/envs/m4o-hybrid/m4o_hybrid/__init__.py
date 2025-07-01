from gym.envs.registration import register
from m4o_hybrid.environments import M4OEnv
import numpy as np

np.random.seed(1)

# data for state
one_data = []
for _ in range(1):
    # default intensity（1000）
    data_cycle = [(np.random.uniform(2.4e9, 4.8e9)) for _ in range(10)]

    # change intensity here
    # data_cycle = [x * 1.75 for x in data_cycle]

    data_cpu = [(np.random.uniform(1.2e9, 1.6e9)) for _ in range(7)]

    # positions and velocities of vehicles
    vehicle_coords = []
    for _ in range(10):
        x = np.random.uniform(-200, 200)
        y = np.random.uniform(-10, 10)
        # velocity of vehicles
        z = np.random.uniform(-20, 20)
        vehicle_coords.extend([x, y, z])

    large_array_ = data_cycle + data_cpu + vehicle_coords
    one_data.append(large_array_)

one_data = np.array(one_data)

# register environment
register(
    id='M4O-v0',
    entry_point='m4o_hybrid:M4OEnv',
    kwargs={
        'one_data':one_data
    }
)