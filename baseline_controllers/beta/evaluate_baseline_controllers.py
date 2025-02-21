'''
# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pystorms'])
subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
'''
import pystorms # this will be the first line of the program when dev is done

import pyswmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import datetime
import os
import swmmio

np.set_printoptions(precision=3,suppress=True)

# BETA
# options are: 'mpc' (per sadler 2020) or 'uncontrolled'
evaluating = 'uncontrolled'
verbose = True
version = "1" # options are "1" and "2"
level = "1" # options are "1" , "2", and "3"
plot = True # plot True significantly increases the memory usage. 
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
# set the random seed
rand_seed = 42
np.random.seed(rand_seed)


print("evaluating ", evaluating, " for beta scenario")

if evaluating == "mpc":
    folder_path = str("./v" + version + "/lev" + level + "/results")
elif evaluating == "uncontrolled":
    folder_path = str("./v" + version + "/results")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# project file is in english units
cfs2cms = 35.315
ft2meters = 3.281



mpc_df = pd.read_csv("mpc_rules_beta.csv")
mpc_df['datetime'] = pd.to_datetime(mpc_df['datetime'])

# convert the dataframe of rules to an easily readable dictionary 
mpc_df['P0'].replace({"ON":1.0, "OFF":0.0}, inplace=True)
mpc_df.drop(columns=["simtime (hr)"], inplace=True)
mpc_datetimes = mpc_df['datetime'].to_list()
mpc_controller = mpc_df.set_index('datetime').to_dict()

# initialize the scenario, and obtain the initial controller settings
#env = pystorms.scenarios.beta(level=level)
env = pystorms.scenarios.beta()
controller_datetime = env.env.sim.start_time
actions = np.array([mpc_controller['R2'][controller_datetime], 
                    mpc_controller['P0'][controller_datetime], 
                    mpc_controller['W0'][controller_datetime]])
done = False

last_eval = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts
last_read = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts

states = pd.DataFrame(columns = env.config['states'])
actions_log = pd.DataFrame(columns = env.config['action_space'])


# Run the simulation
while not done:

    if evaluating == "mpc" and (env.env.sim.current_time >= controller_datetime) and (controller_datetime in mpc_datetimes):
        actions = np.array([mpc_controller['R2'][controller_datetime], 
                            mpc_controller['P0'][controller_datetime], 
                            mpc_controller['W0'][controller_datetime]])
        controller_datetime += datetime.timedelta(minutes=15)
    elif evaluating == "uncontrolled":
        actions = np.array([1.0, 0.0, 1.0]) # orifice and weir open, pump off.
    else:
        exit(1)
    if (not done) and verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 4 == 0: 
        u_print = actions.flatten()
        y_measured = env.state(level=level).reshape(-1,1)
        # print the names of the states and their current values side-by-side
        print("state, value")
        for idx in range(len(env.config['states'])):
            print(env.config['states'][idx], y_measured[idx])
        print("actuator, setting")
        for idx in range(len(env.config['action_space'])):
            print(env.config['action_space'][idx], u_print[idx])

        print("current time, end time")
        print(env.env.sim.current_time, env.env.sim.end_time)
        print("\n")
            
    if (not done) and (env.env.sim.current_time > (last_read + datetime.timedelta(minutes=1))) and plot: # log data 
        last_read = env.env.sim.current_time
        state = env.state(level="1").reshape(1,len(env.config['states']))
        current_state = pd.DataFrame(data=state, columns = env.config['states'], index = [env.env.sim.current_time] )
        states = pd.concat((states,current_state))
        action = actions.reshape(1,len(env.config['action_space']))
        current_actions = pd.DataFrame(data = action, columns = env.config['action_space'], index=[env.env.sim.current_time])
        actions_log = pd.concat((actions_log, current_actions))
            

    if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
        final_depths = env.state(level="1")

    #done = env.step(actions.flatten(),level=level)
    done = env.step(actions.flatten())
    # note that noise won't affect the controller as it's predetermined
    # but actuator faults such as getting stuck will still affect the controller
perf = sum(env.data_log["performance_measure"])
print("flooding")
for key,value in env.data_log['flooding'].items():
    # flooded nodes
    if sum(value) > 0:
        print(key, sum(value))
        

# save the cost and ending dpeths to a csv
perf_summary = pd.DataFrame(data = {"cost": perf, "final_depths": final_depths})
if evaluating == "uncontrolled":
    perf_summary.to_csv(str(folder_path + "/costs_" + evaluating + ".csv"))
else:
    perf_summary.to_csv(str(folder_path + "/costs_" + evaluating +  ".csv"))

if plot:
        
    if evaluating == "uncontrolled":
        states.to_csv(str(folder_path + "/states_" + str(evaluating) + ".csv"))
        actions_log.to_csv(str(folder_path + "/actions_" + str(evaluating) + ".csv"))
        # and the data log
        with open(f'./v{version}/results/{evaluating}_data_log.pkl', 'wb') as f:
            pickle.dump(env.data_log, f)
    else:
        # save the flows and depths
        states.to_csv(str(folder_path + "/states_" + str(evaluating) + ".csv"))
        actions_log.to_csv(str(folder_path + "/actions_" + str(evaluating) + ".csv"))
        # and the data log
        with open(f'./v{version}/lev{level}/results/{str(evaluating )}_data_log.pkl', 'wb') as f:
            pickle.dump(env.data_log, f)
        
        
    # if there are any na values in states or weir_heads32, linearly interpolate over them
    states.interpolate(method='time',axis='index',inplace=True)
    #weir_heads32.interpolate(method='time',axis='index',inplace=True)
    actions_log.interpolate(method='time',axis='index',inplace=True)
    #flows.interpolate(method='time',axis='index',inplace=True)

    plots_high = max(len(env.config['action_space']) , len(env.config['states']))
    fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

    axes[0,0].set_title("actions")
    axes[0,1].set_title("states")
    # plot the actions
    for idx in range(len(env.config['action_space'])):
        axes[idx,0].plot(actions_log.iloc[:,idx])
        axes[idx,0].set_ylabel("setting")
        if idx == len(env.config['action_space']) - 1:
            axes[idx,0].set_xlabel("time")
            # plot only the first, middle, and last x-ticks
            xticks = axes[idx,0].get_xticks()
            xticks = [xticks[0],xticks[int(len(xticks)/2)],xticks[-1]]
            axes[idx,0].set_xticks(xticks)
        if idx != len(env.config['action_space']) - 1: # not the last row
            axes[idx,0].set_xticklabels([])
        axes[idx,0].annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
        

    # plot the states
    for idx in range(len(env.config['states'])):
        axes[idx,1].plot(states.iloc[:,idx])

        if idx == len(env.config['states']) - 1:
            axes[idx,1].set_xlabel("time")
            axes[idx,1].annotate(str(env.config['states'][idx]), xy=(0.5, 0.4), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
            # plot only the first, middle, and last x-ticks
            xticks = axes[idx,1].get_xticks()
            xticks = [xticks[0],xticks[int(len(xticks)/2)],xticks[-1]]
            axes[idx,1].set_xticks(xticks)

        if idx != len(env.config['states']) - 1:
            axes[idx,1].set_xticklabels([])
            axes[idx,1].annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')


    plt.tight_layout()
    if evaluating == "uncontrolled":
        plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + ".png"),dpi=450)
        plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + ".svg"),dpi=450)
    else:
        plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) +  ".png"),dpi=450)
        plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) +  ".svg"),dpi=450)
    #plt.show()
    plt.close('all')