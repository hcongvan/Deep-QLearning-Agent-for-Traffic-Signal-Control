import sys,os
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import numpy as np
import random
import matplotlib.pyplot as plt

from sumolib import checkBinary
import TrafficGenerator
from TrafficGenerator import TrafficGenerator
# sumo things - we need to import python modules from the $SUMO_HOME/tools directory


def _get_waiting_times():
    incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    _waiting_times = {}
    for veh_id in traci.vehicle.getIDList():
        wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
        if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
            _waiting_times[veh_id] = wait_time_car
        else:
            if veh_id in _waiting_times:
                del _waiting_times[veh_id]  # the car isnt in incoming roads anymore, delete his waiting time
    total_waiting_time = sum(_waiting_times.values())
    return total_waiting_time

sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_test.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(100),"--start","--quit-on-end"]
traci.start(sumoCmd)
max_step = 5000
_waiting_times = {}
waitTime_per_step = []
vehicle_per_step = []
for i in range(max_step):
    traci.simulationStep()
    waitTime_per_step.append(_get_waiting_times())
    vehicle_per_step.append(len(traci.vehicle.getIDList()))
plot_path = "./model/default/"
# wait Time per step
data = waitTime_per_step
plt.plot(data)
plt.ylabel("Total Waiting Time")
plt.xlabel("Step")
plt.margins(0)
min_val = min(data)
max_val = max(data)
plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
fig = plt.gcf()
fig.set_size_inches(20, 11.25)
fig.savefig(plot_path + 'waitTime_per_step.png', dpi=96)
plt.close("all")
with open(plot_path + 'waitTime_per_step.txt', "w") as file:
    for item in data:
            file.write("%s\n" % item)

# vehicle per step
data = vehicle_per_step
plt.plot(data)
plt.ylabel("Vehicle")
plt.xlabel("Step")
plt.margins(0)
min_val = min(data)
max_val = max(data)
plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
fig = plt.gcf()
fig.set_size_inches(20, 11.25)
fig.savefig(plot_path + 'vehicle_per_step.png', dpi=96)
plt.close("all")
with open(plot_path + 'vehicle_per_step.txt', "w") as file:
    for item in data:
            file.write("%s\n" % item)
