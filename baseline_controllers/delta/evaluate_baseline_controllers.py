'''
# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
from tkinter import Y
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pystorms'])
subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
'''
import pystorms # this will be the first line of the program when dev is done
import copy
import pyswmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import datetime
import os
import swmmio

np.set_printoptions(precision=3,suppress=True)

# DELTA
# options are: 'static-plus-rule' and 'prop-outflow' (or 'uncontrolled')
# static plus rule is fixed positions for the 4 weirs plus a height to open the infiltration valve at
# prop outflow adds proportional feedback to weir openings to increase outflows when far below the threshold
evaluating = 'static-plus-rule' 
verbose = True
version = "2" # options are "1" and "2"
level = "1" # options are "1" , "2", and "3"
#hysteresis = 0.05 # hysteresis for the infiltration valve to avoid rapid cycling between open and closed
plot = True # plot True significantly increases the memory usage. 
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
# set the random seed
rand_seed = 7
np.random.seed(rand_seed)

print("evaluating ", evaluating, " for delta scenario")

tuning_values = np.arange(-0.05,0.25,0.01)
tuning_values = np.round(tuning_values,2)

# for dev or plotting - single value
tuning_values = [0.0]

if evaluating == "uncontrolled":
    folder_path = str("./v" + version + "/results")
else:
    folder_path = str("./v" + version + "/lev" + level + "/results")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

operational_bounds_data = {
    "Lower Limit": [5.32, 4.44, 5.20, 3.28, None],
    "Upper Limit": [5.92, 5.04, 5.80, 3.80, 6.55]
}

operational_bounds_df = pd.DataFrame(operational_bounds_data, index=["N3", "N2", "N1", "C", "S"])

exceedance_bounds_data = {
    "Lower Limit": [5.28, 4.04, 2.11, 2.21, None],
    "Upper Limit": [11.99, 6.59, 5.92, 5.70, 9.55]
}

exceedance_bounds_df = pd.DataFrame(exceedance_bounds_data, index=["N3", "N2", "N1", "C", "S"])
    

for parameter in tuning_values:
    if evaluating == "static-plus-rule":
        optimal_static_settings = np.loadtxt(str("./v" + version + "/optimal_static.txt"))
        #optimal_static_settings = np.array([0.95,0.3,0.11,0.47,3.73,3.93])
    elif evaluating == "prop-outflow":
        optimal_static_settings = np.loadtxt(str("./v" + version + "/optimal_prop.txt"))[:-1]
        optimal_prop = np.loadtxt(str("./v" + version + "/optimal_prop.txt"))[-1]
    elif evaluating == "uncontrolled":
        optimal_static_settings = np.loadtxt(str("./v" + version + "/optimal_static.txt"))
        optimal_static_settings[-2] = 0.05 # infiltration valve always a little open

    print("tuning value: ", parameter)
    optimal_static_settings = optimal_static_settings*(1+parameter)
    env = pystorms.scenarios.delta(version=version,level=level)

    env.env.sim.start()
    done = False
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    #u_open_pct = constant_flows
    # make u_open_pct a deep copy of constant_flows (constant_flows should not change)
    #u_open_pct = copy.deepcopy(optimal_static_settings)
    u_open_pct = copy.deepcopy(optimal_static_settings[:-1])
    # start with valve closed
    

    states = pd.DataFrame(columns = env.config['states'])
    actions = pd.DataFrame(columns = env.config['action_space'])
    
    while not done:
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state(level=level)
            
            # first bit is the same for all 3 controllers
            # set the weirs
            #u_open_pct[:-1] = optimal_static_settings[:-1]
            u_open_pct[:-1] = optimal_static_settings[:-2]
            # open valve?
            if evaluating != "uncontrolled":
                '''
                if state[0] > optimal_static_settings[-1]: # basin c depth above optimized threshold?
                    u_open_pct[-1] = 1.0 # fully open valve above threshold depth in basin c
                else:
                    u_open_pct[-1] = 0.0 # close the valve to preserve capacity in the infiltration basin    
                '''
                # try proprtional valve opening between two setpoint depths
                if state[0] < optimal_static_settings[-2]: # below lower threshold
                    u_open_pct[-1] = 0.0 # close the valve to preserve capacity in the infiltration basin
                elif state[0] > optimal_static_settings[-1]: # above upper threshold
                    u_open_pct[-1] = 1.0 # fully open valve above upper threshold depth in basin c
                else: # between the two thresholds
                    # linearly interpolate the valve opening based on the current depth in basin C
                    # find the range between the two thresholds
                    lower_threshold = optimal_static_settings[-2] # lower threshold depth
                    upper_threshold = optimal_static_settings[-1] # upper threshold depth
                    range_threshold = upper_threshold - lower_threshold
                    # calculate the current depth in the range
                    current_depth_in_range = state[0] - lower_threshold
                    # calculate the percentage of the way through the range
                    percentage_in_range = current_depth_in_range / range_threshold
                    # set the valve opening based on the percentage in range
                    u_open_pct[-1] = percentage_in_range # linearly interpolate the valve opening based on the current depth in basin C
                            

            if evaluating == "prop-outflow":
                # how much are outflows below the threshold?
                current_outflow = pyswmm.Links(env.env.sim)['conduit_Edown'].flow
                flow_capacity = env.threshold - current_outflow
                # flow capacity may be negative, in which case the weirs would raise to reduce outflows

                for idx, name_w_weir_prefix in enumerate(env.config['action_space'][:-1]):
                    name = name_w_weir_prefix[5:]
                    # find the index of the entry in "states" which contains that name. it will contain it but not be an exact match
                    idx_state = [i for i, s in enumerate(states.columns) if name in s[0]]
                    op_bound_width = operational_bounds_df.loc[name,"Upper Limit"] - operational_bounds_df.loc[name, "Lower Limit"]
                    lower_tight_bound = operational_bounds_df.loc[name, "Lower Limit"] + 0.1 * op_bound_width
                    upper_tight_bound = operational_bounds_df.loc[name,"Upper Limit"] - 0.25 * op_bound_width
                    if state[idx_state[0]] < upper_tight_bound and state[idx_state[0]] > lower_tight_bound:
                        # only use proportional feedback on the outflow if we're safe regarding depth bounds
                        u_open_pct[idx] += optimal_prop * flow_capacity
                    elif state[idx_state[0]] >= upper_tight_bound and optimal_prop*flow_capacity > 0: # allow extra opening if close to exceeding upper bound
                        u_open_pct[idx] += optimal_prop * flow_capacity
                    if u_open_pct[idx] > 1.0:
                        u_open_pct[idx] = 1.0
                    elif u_open_pct[idx] < 0.0:
                        u_open_pct[idx] = 0.0
                    

        if (not done) and verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 4 == 0: 
                u_print = u_open_pct.flatten()
                y_measured = env.state().reshape(-1,1)
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
            
        if (not done) and (env.env.sim.current_time > last_read + datetime.timedelta(minutes=1)) and plot: # log data 
            last_read = env.env.sim.current_time
            state = env.state().reshape(1,len(env.config['states']))
            current_state = pd.DataFrame(data=state, columns = env.config['states'], index = [env.env.sim.current_time] )
            states = pd.concat((states,current_state))
            action = u_open_pct.reshape(1,len(env.config['action_space']))
            current_actions = pd.DataFrame(data = action, columns = env.config['action_space'], index=[env.env.sim.current_time])
            actions = pd.concat((actions, current_actions))

        if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
            node_indices = [i for i in range(len(env.config['states'])) if 'depthN' in env.config['states'][i][1]]
            final_depths = env.state()[node_indices]
    
        done = env.step(u_open_pct.flatten(),level=level)
        
    perf = sum(env.data_log["performance_measure"])
    print("performance of: ", evaluating)
    print("{:.4e}".format(perf))
    flood_cost = 0
    for key,value in env.data_log['flooding'].items():
        if sum(value) > 0:
            # the total cost is the number of positive entries in value multiplied by the cost of flooding (10e6)
            # find the number of positive entries in value
            num_positive = sum([1 for x in value if x > 0])
            print(key,"{:.4e}".format(num_positive*(10**6)))
            flood_cost += num_positive*(10**6)
    print("total flood cost:","{:.4e}".format(flood_cost))
    excess_flows = [x - env.threshold for x in env.data_log['flow']['conduit_Eout']] 
    excess_flows = [x if x > 0 else 0 for x in excess_flows]
    flow_cost = sum(excess_flows)*10
    print("total flow cost (leaving network exceeds threshold):","{:.4e}".format(flow_cost))
    #print("depth bounds cost:","{:.4e}".format(perf - flood_cost - flow_cost))
    op_bounds_cost = 0.0
    for key,value in env.data_log['depthN'].items():
        exceed_upper = [x - operational_bounds_df.loc[key[6:],"Upper Limit"] for x in value]
        exceed_upper = [x if x > 0 else 0 for x in exceed_upper]
        exceed_lower = [x - operational_bounds_df.loc[key[6:],"Lower Limit"] for x in value]
        exceed_lower = [x if x < 0 else 0 for x in exceed_lower]
        op_bounds_cost += np.abs(sum(exceed_upper)) + np.abs(sum(exceed_lower))
    exceedance_cost = 0.0
    for key,value in env.data_log['depthN'].items():
        exceed_upper = [x - exceedance_bounds_df.loc[key[6:],"Upper Limit"] for x in value]
        exceed_upper = [x if x > 0 else 0 for x in exceed_upper]
        exceed_lower = [x - exceedance_bounds_df.loc[key[6:],"Lower Limit"] for x in value]
        exceed_lower = [x if x < 0 else 0 for x in exceed_lower]
        exceedance_cost += np.abs(sum(exceed_upper)) + np.abs(sum(exceed_lower))
    # flood cost and exceedance cost will define feasibility
    # if exceedance cost is zero, record 1 - (closest approach to exceedance bounds)/(closest intitial distance from exceedance bounds)
    # if we never get closer to the exceedance bounds than our initial depths, then this records 0
    # otherwise it scales up to 1.0 as we barely avoid exceeding the bounds, normalized by our initial distance
    if exceedance_cost == 0.0:
        closest_initial_distance = np.inf
        closest_approach = np.inf
        for key,value in env.data_log['depthN'].items():
            initial_distance = np.abs(value[0] - exceedance_bounds_df.loc[key[6:],"Lower Limit"])
            if initial_distance < closest_initial_distance:
                closest_initial_distance = initial_distance
            initial_distance = np.abs(exceedance_bounds_df.loc[key[6:], "Upper Limit"] - value[0])
            if initial_distance < closest_initial_distance:
                closest_initial_distance = initial_distance

            approach = np.abs(np.array(value) - exceedance_bounds_df.loc[key[6:],"Lower Limit"])
            if min(approach) < closest_approach:
                closest_approach = min(approach)
            approach = np.abs(np.array(value) - exceedance_bounds_df.loc[key[6:],"Upper Limit"])
            if min(approach) < closest_approach:
                closest_approach = min(approach)
        exceedance_cost = 1 - (closest_approach/closest_initial_distance)
    print("operational bounds cost:","{:.4e}".format(op_bounds_cost))
    print("exceedance cost:","{:.4e}".format(exceedance_cost))

    # round final_depths to 2 decimal places
    final_depths = np.round(final_depths,2)
    # calcuate the final filling degrees of all measured depths
    #final_filling_degrees = np.round(final_depths / max_depths_array , 2)
    # save the cost and ending filling degree to a csv
    perf_summary = pd.DataFrame(data = {"cost": perf, 
                                        "final_depths": final_depths,"flow_cost":flow_cost,
                                        "op_bounds_cost":op_bounds_cost, "exceedance_cost":exceedance_cost,
                                        "flood_cost":flood_cost})
    if evaluating == "uncontrolled":
        perf_summary.to_csv(str(folder_path + "/costs_" + evaluating + ".csv"))
    else:
        perf_summary.to_csv(str(folder_path + "/costs_" + evaluating + "_a=" + str(parameter) + ".csv"))


    if plot:
        
        # if there are any na values in states or weir_heads32, linearly interpolate over them
        states.interpolate(method='time',axis='index',inplace=True)
        #weir_heads32.interpolate(method='time',axis='index',inplace=True)
        actions.interpolate(method='time',axis='index',inplace=True)

        plots_high = max(len(env.config['action_space']) , len(env.config['states']))
        fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

        axes[0,0].set_title("actions")
        axes[0,1].set_title("states")
        # plot the actions
        for idx in range(len(env.config['action_space'])):
            #axes[idx,0].plot(weir_heads32.iloc[:,idx])
            #axes[idx,0].set_ylabel("(weir head [ft])^(3/2)")
            axes[idx,0].plot(actions.iloc[:,idx])
            axes[idx,0].set_ylabel("fraction open")
            if idx == len(env.config['action_space']) - 1:
                axes[idx,0].set_xlabel("time")
                # plot only the first, middle, and last x-ticks
                xticks = axes[idx,0].get_xticks()
                xticks = [xticks[0],xticks[int(len(xticks)/2)],xticks[-1]]
                axes[idx,0].set_xticks(xticks)
            if idx != len(env.config['action_space']) - 1: # not the last row
                axes[idx,0].set_xticklabels([])
            axes[idx,0].annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
        # plot flows through Eout
        axes[-1,0].plot(env.data_log['flow']['conduit_Eout'])
        axes[-1,0].set_ylabel("flow")
        axes[-1,0].set_xlabel("time")
        # add a dotted red line at threshold
        axes[-1,0].axhline(y=env.threshold, color='r', linestyle='dotted')
        # plot the states
        for idx, state in enumerate(env.config['states']):
            axes[idx,1].plot(states.iloc[:,idx])
            # add bound lines (dotted blue operational, solid red exceedance)
            if '4' not in state[0]: # basin N4 isn't controlled and doesn't have bounds
                if "S" not in state[0]: # basin S doesn't have lower bounds defined
                    axes[idx,1].axhline(y=operational_bounds_df.loc[state[0][6:],'Lower Limit']  , color='b', linestyle='dotted')
                    axes[idx,1].axhline(y=exceedance_bounds_df.loc[state[0][6:],'Lower Limit']  , color='r')
                
                axes[idx,1].axhline(y=operational_bounds_df.loc[state[0][6:],'Upper Limit']  , color='b', linestyle='dotted')
                axes[idx,1].axhline(y=exceedance_bounds_df.loc[state[0][6:],'Upper Limit']  , color='r')

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
            plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + "_param=" + str(parameter) + ".png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + "_param=" + str(parameter) + ".svg"),dpi=450)
        #plt.show()
        plt.close('all')
        
        states = states.resample('5min').mean()
        actions = actions.resample('5min').mean()

        if evaluating == "uncontrolled":
            states.to_csv(str(folder_path + "/states_" + str(evaluating) + ".csv"))
            actions.to_csv(str(folder_path + "/actions_" + str(evaluating) + ".csv"))
            # and the data log
            with open(f'./v{version}/results/{evaluating}_data_log.pkl', 'wb') as f:
                pickle.dump(env.data_log, f)
        else:
            # save the flows and depths
            states.to_csv(str(folder_path + "/states_" + str(evaluating) + "_param=" + str(parameter) + ".csv"))
            actions.to_csv(str(folder_path + "/actions_" + str(evaluating) + "_param=" + str(parameter) + ".csv"))
            # and the data log
            with open(f'./v{version}/lev{level}/results/{str(evaluating + "_param=" + str(parameter))}_data_log.pkl', 'wb') as f:
                pickle.dump(env.data_log, f)
