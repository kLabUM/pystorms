# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pystorms'])
subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])

import pystorms # this will be the first line of the program when dev is done

import pyswmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import datetime
import os

np.set_printoptions(precision=3,suppress=True)

# options are: 'equal-filling' and 'constant-flow'
control_scenario = 'equal-filling' 
verbose = True
version = "2" # options are "1" and "2"
level = "3" # options are "1" , "2", and "3"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
# set the random seed
rand_seed = 42
np.random.seed(rand_seed)

optimal_constant_flows = np.loadtxt(str("./" + version + "/optimal_constant_flows.txt"))
optimal_efd_params = np.loadtxt(str("./" + version + "/optimal_efd_params.txt"))

print("evaluating ", control_scenario, " for theta scenario")

Cd = 1.0 # same for both valves
Ao = 1 # area is one square meter
g = 9.81 # m / s^2


env = pystorms.scenarios.theta(level=level)
env.env.sim.start()
done = False

max_depths_array = np.array([])
max_depths = dict()
for state in env.config['states']:
    if 'depth' in state[1]:
        node_id = state[0]
        max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
        max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)


u = np.ones((len(env.config['action_space']),1)) # begin all open
u_open_pct = np.ones((len(env.config['action_space']),1)) # begin all open

depths = pd.DataFrame(columns = env.config['states'])
flows = pd.DataFrame(columns = env.config['action_space'])


last_eval = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts
last_read = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts

print(env.env.actuator_schedule)
print(env.env.sensor_schedule)

while not done:

    # take control actions?
    if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
        state = env.state(level=level)
        #print("clean," , env.state(level="1"))
        #print("level 2," , env.state(level="2"))
        #print("level 3," , env.state(level="3"))
        if control_scenario == 'equal-filling' or control_scenario == 'constant-flow':
            
            for idx in range(len(u_open_pct)): # set opening percentage to achieve the desired flow rate
       
                # flow rate for an orifice is Q = CA sqrt(2gh)
                # assume this scales linearly with opening percentage
                Q_desired = optimal_constant_flows[idx]
                if state[idx] < 1e-3:
                    u_open_pct[idx,0] = 1.0
                else:
                    u_open_pct[idx,0] = Q_desired / (Cd*Ao*np.sqrt(2*g*state[idx]))
                # bound the opening percentage to [0,1]
                if u_open_pct[idx,0] > 1:
                    u_open_pct[idx,0] = 1
                elif u_open_pct[idx,0] < 0:
                    u_open_pct[idx,0] = 0
                
            if control_scenario == "equal-filling":
                filling_degrees = np.array([pyswmm.Nodes(env.env.sim)[node_id].depth/max_depths[node_id] for node_id in max_depths.keys()]).reshape(-1,1)
                u_diff = filling_degrees - np.mean(filling_degrees) 
                u_open_pct = u_open_pct + optimal_efd_params*u_diff
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0] > 1:
                        u_open_pct[i,0] = 1
                    elif u_open_pct[i,0] < 0:
                        u_open_pct[i,0] = 0

        
        if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0:
            y_measured = state.reshape(-1,1)
            print("u_open_pct, y_measured")
            print(np.c_[u_open_pct,y_measured])
            print("current time, end time")
            print(env.env.sim.current_time, env.env.sim.end_time)
            print("\n")

    if env.env.sim.current_time > last_read + datetime.timedelta(minutes=1): # log data 
        last_read = env.env.sim.current_time
        # logging data since pystorms doesn't log the observables and controllables for this scenario
     
        state = env.state().reshape(1,len(env.config['states']))
        current_depths = pd.DataFrame(data=state, columns = env.config['states'], index = [env.env.sim.current_time] )
        depths = pd.concat((depths,current_depths))
        current_actions = pd.DataFrame(data=u_open_pct.reshape(1,len(env.config['action_space'])), 
                                       columns=env.config['action_space'], index = [env.env.sim.current_time] )
        #current_flows = Cd*Ao*current_actions.values*np.sqrt(2*g*current_depths.values)
        Links = pyswmm.Links(env.env.sim)
        current_flows = np.array([Links['7'].flow, Links['9'].flow]).reshape(1,2)
        flows = pd.concat((flows,pd.DataFrame(data=current_flows, columns=env.config['action_space'], index = [env.env.sim.current_time])),axis=0)
    
    
    done = env.step(u_open_pct.flatten(),level=level)


perf = sum(env.data_log["performance_measure"])
print("performance:")
print("{:.4e}".format(perf))


plots_high = max(len(env.config['action_space']) , len(env.config['states']))
fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

axes[0,0].set_title("flows")
axes[0,1].set_title("depths")


# plot the flows
for idx in range(len(env.config['action_space'])):
    axes[idx,0].plot(flows.iloc[:,idx], label=env.config['action_space'][idx])
    #axes[idx,0].axhline(y=flow_threshold, color='r', linestyle='-')
    axes[idx,0].set_ylabel("flow (cms)")
    axes[idx,0].set_xlabel("time")
    #axes[idx,0].legend()
    if idx != len(env.config['action_space']) - 1: # not the last row
        axes[idx,0].set_xticklabels([])
    axes[idx,0].annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the depths
for idx in range(len(env.config['states'])):
    axes[idx,1].plot(depths.iloc[:,idx], label=env.config['states'][idx])
    #axes[idx,1].axhline(y=depth_threshold, color='r', linestyle='-')
    axes[idx,1].set_ylabel("depth (m)")
    axes[idx,1].set_xlabel("time")
    #axes[idx,1].legend()
    if idx != len(env.config['states']) - 1:
        axes[idx,1].set_xticklabels([])
    axes[idx,1].annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

plt.tight_layout()
plt.savefig(str("./" + version +"/evaluate_" + str(control_scenario) + "_level_" + level + ".png"),dpi=450)
plt.savefig(str("./" + version +"/evaluate_" + str(control_scenario) + "_level_" + level + ".svg"),dpi=450)
#plt.show(block=True)
plt.close('all')


# resample the flows and depths to 5 minute intervals
flows = flows.resample('5T').mean()
depths = depths.resample('5T').mean()

# save the flows and depths
flows.to_csv(str("./" + version +"/flows_" + str(control_scenario) + "_level_" + level +  ".csv"))
depths.to_csv(str("./" + version +"/depths_" + str(control_scenario) + "_level_" + level +  ".csv"))
# and the data log
with open(f'./{version}/{control_scenario}_level_{level}_data_log.pkl', 'wb') as f:
            pickle.dump(env.data_log, f)
            