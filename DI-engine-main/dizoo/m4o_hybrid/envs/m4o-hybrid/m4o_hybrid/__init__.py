from gym.envs.registration import register
from m4o_hybrid.environments import M4OEnv
import numpy as np
import traci
import time
import random

# generate traffic flow file

def generate_flow_file():
    output_file = r"D:\SUMO\hppo\flow_cross.rou.xml"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        f.write('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')
        f.write('    <vType id="yellowCar" accel="2.6" decel="4.5" sigma="0.5"\n')
        f.write('           length="4.5" minGap="0.1" maxSpeed="20" color="1,1,0" guiShape="passenger"/>\n\n')
        f.write('    <vType id="redCar" accel="2.6" decel="4.5" sigma="0.5"\n')
        f.write('           length="4.5" minGap="0.1" maxSpeed="20" color="1,0,0" guiShape="passenger"/>\n\n')

        f.write(' <route id="right_to_left" edges="E1 -E2"/>\n')
        f.write(' <route id="left_to_right" edges="E2 -E1"/>\n\n')

        f.write(' <routeDistribution id="route_random">\n')
        f.write('     <route refId="right_to_left" probability="0.5"/>\n')
        f.write('     <route refId="left_to_right" probability="0.5"/>\n')
        f.write(' </routeDistribution>\n\n')

        # redCar - CV, yellowCar - SV
        for i in range(1, 11):
            if i <= 3:
                v_type = "redCar"
            else:
                v_type = "yellowCar"

            t_speed = round(random.uniform(0, 20), 2)  # target speed: U(0,20)

            f.write(f'    <vehicle id="car{i}" type="{v_type}" route="route_random" depart="0"\n')
            f.write('             departLane="random" departPos="random_free" departSpeed="random">\n')
            f.write(f'        <param key="targetSpeed" value="{t_speed}"/>\n')
            f.write('    </vehicle>\n\n')

        f.write('</routes>\n')


# start SUMO

def start_sumo():
    sumo_cmd = ["sumo", "-c", r"D:\SUMO\hppo\m4o.sumocfg", "--start", "--quit-on-end"]
    traci.start(sumo_cmd)

# get vehicle states

def read_vehicle_states():
    time.sleep(0.5)  # wait for SUMO to finish loading

    traci.simulationStep()

    veh_ids = traci.vehicle.getIDList()[:10]

    # Get initial vehicle positions and speeds
    xs, ys, speeds = [], [], []

    for vid in veh_ids:
        x, y = traci.vehicle.getPosition(vid)
        v = traci.vehicle.getSpeed(vid)
        xs.append(x)
        ys.append(y)
        speeds.append(v)

    return xs, ys, speeds

# construct one_data for training

def build_one_data(xs, ys, speeds):
    vehicle_coords = []
    for i in range(10):
        vehicle_coords += [xs[i], ys[i], speeds[i]]

    # computational intensity (1000)
    data_bit = [np.random.uniform(2.4e9, 4.8e9) for _ in range(10)]
    # change intensity here
    # data_bit = [x * 2 for x in data_bit]

    data_cpu = [np.random.uniform(1.2e9, 1.6e9) for _ in range(7)]

    return np.array([data_bit + data_cpu + vehicle_coords])


# Main process

generate_flow_file()
start_sumo()

xs, ys, speeds = read_vehicle_states()
one_data = build_one_data(xs, ys, speeds)

# register environment
register(
    id='M4O-v0',
    entry_point='m4o_hybrid:M4OEnv',
    kwargs={
        'one_data':one_data
    }
)
