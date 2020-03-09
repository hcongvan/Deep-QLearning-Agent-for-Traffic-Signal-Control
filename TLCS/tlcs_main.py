# created by Andrea Vidali
# info@andreavidali.com

from __future__ import absolute_import
from __future__ import print_function
import ctypes
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cufft64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\curand64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cusolver64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cusparse64_100.dll")
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudnn64_7.dll")
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import numpy as np
import math
import timeit
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from SimRunner import SimRunner
from TrafficGenerator import TrafficGenerator
from Memory import Memory
from Model import Model

# sumo things - we need to import python modules from the $SUMO_HOME/tools directory

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def save_graphs(sim_runner, total_episodes, plot_path):

    plt.rcParams.update({'font.size': 24})  # set bigger font size

    # reward
    data = sim_runner.reward_store
    plt.plot(data)
    plt.ylabel("Cumulative negative reward")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'reward_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

    # cumulative wait
    data = sim_runner.cumulative_wait_store
    plt.plot(data)
    plt.ylabel("Cumulative delay (s)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'delay_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)

    # average number of cars in queue
    data = sim_runner.avg_intersection_queue_store
    plt.plot(data)
    plt.ylabel("Average queue length (vehicles)")
    plt.xlabel("Episode")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue.png', dpi=96)
    plt.close("all")
    with open(plot_path + 'queue_data.txt', "w") as file:
        for item in data:
                file.write("%s\n" % item)



if __name__ == "__main__":

    # --- TRAINING OPTIONS ---
    gui = True
    total_episodes = 10
    gamma = 0.75
    batch_size = 10
    memory_size = 50000
    path = "./model/model_1_5x400_100e_075g/"  # nn = 5x400, episodes = 300, gamma = 0.75
    test = True
    # ----------------------

    # attributes of the agent
    num_states = 80
    num_actions = 4
    max_steps = 100  # seconds = 1 h 30 min each episode
    green_duration = 10
    yellow_duration = 4
    image_shape = (224,224,3)

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # initializations
    model = Model(image_shape,num_states, num_actions, batch_size)
    memory = Memory(memory_size)
    traffic_gen = TrafficGenerator(max_steps)
    if test:
        sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_test.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps),"--start","--quit-on-end"]
    else:
        sumoCmd = [sumoBinary, "-c", "intersection/tlcs_config_train.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps),"--start","--quit-on-end"]
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("PATH:", path)
        print("----- Start time:", datetime.datetime.now())
        sess.run(model.var_init)
        sim_runner = SimRunner(sess, model, memory, traffic_gen, total_episodes, gamma, max_steps, green_duration, yellow_duration, sumoCmd, test, path)
        episode = 0

        while episode < total_episodes:
            print('----- Episode {} of {}'.format(episode+1, total_episodes))
            start = timeit.default_timer()
            sim_runner.run(episode)  # run the simulation
            stop = timeit.default_timer()
            print('Time: ', round(stop - start, 1))
            episode += 1
        if not test:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            saver.save(sess, path + "my_tlcs_model.ckpt") 
            print("----- End time:", datetime.datetime.now())
            print("PATH:", path)
        save_graphs(sim_runner, total_episodes, path)
