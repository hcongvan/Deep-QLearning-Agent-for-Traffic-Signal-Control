import traci
import numpy as np
import random
import sys,os
from sumolib import checkBinary
import TrafficGenerator
from TrafficGenerator import TrafficGenerator
# sumo things - we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


sumoBinary = checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(100),"--start"]
traci.start(sumoCmd)
durations = 10
for i in range(durations):
    traci.simulationStep()
    if i % 10 == 0:
        traci.gui.screenshot('View #0','image/{}.png'.format(str(i)))
