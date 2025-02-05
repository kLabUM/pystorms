'''
# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
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

# GAMMA
# options are: 'equal-filling' and 'constant-flow' (or 'uncontrolled')
evaluating = 'equal-filling' 
verbose = True
version = "2" # options are "1" and "2"
level = "1" # options are "1" , "2", and "3"
plot = True # plot True significantly increases the memory usage. 
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
# set the random seed
rand_seed = 42
np.random.seed(rand_seed)

print("evaluating ", evaluating, " for gamma scenario")

tuning_values = np.arange(-0.05,0.25,0.01)
tuning_values = np.round(tuning_values,2)

# for dev or plotting - single value
tuning_values = [0.0]

if evaluating == "constant-flow" or evaluating == "equal-filling":
    folder_path = str("./v" + version + "/lev" + level + "/results")
elif evaluating == "uncontrolled":
    folder_path = str("./v" + version + "/results")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    

for parameter in tuning_values:
    #optimal_constant_flows = np.loadtxt(str("./v" + version + "/optimal_constant_flows.txt"))
    optimal_constant_flows = np.ones(9)*6.0
    #optimal_constant_flows[[4,8]] = 75.0 # 5 and 9 will flood
    #optimal_efd_params = np.loadtxt(str("./v" + version + "/optimal_efd_params.txt"))
    optimal_efd_params = 0.5

    print("tuning value: ", parameter)
    optimal_constant_flows = optimal_constant_flows*(1+parameter)
    
    # gamma is in imperial units of feet and cubic feet per second
    cfs2cms = 35.315
    ft2meters = 3.281
    #env = pystorms.scenarios.gamma(version=version,level=level)
    env = pystorms.scenarios.gamma()
    for state in env.config['states']:
        # if state contains 5 or 9, remove it
        if '5' in state[0] or '9' in state[0]:
            env.config['states'].remove(state)
    for action in env.config['action_space']:
        # if action contains 5 or 9, remove it
        if '5' in action or '9' in action:
            env.config['action_space'].remove(action)
    for target in env.config['performance_targets']:
        # if target contains 5 or 9, remove it
        if '5' in target[0] or '9' in target[0]:
            env.config['performance_targets'].remove(target)
    #print(env.config['states'])
    #print(env.config['action_space'])
    #print(env.config['performance_targets'])
    env.env.sim.start()
    done = False
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    
    u_open_pct = 0.0*np.ones(len(env.config['action_space'])) # start closed

    states = pd.DataFrame(columns = env.config['states'])
    actions = pd.DataFrame(columns = env.config['action_space'])
    flows = pd.DataFrame(columns = env.config['action_space'])

    #print(env.env.actuator_schedule)
    #print(env.env.sensor_schedule)

    max_depths_array = np.array([]) # for all states
    for state in env.config['states']:
        if 'depthN' in state[1]:
            node_id = state[0]
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)


    model = swmmio.Model(env.config["swmm_input"])
    #print(model.inp.orifices)
    orifice_areas = dict()
    for ori in model.inp.orifices.index.tolist():
        # Geom1 and geom2 are the height and width of the orifice
        orifice_areas[ori] = model.inp.xsections.loc[ori, 'Geom1']*model.inp.xsections.loc[ori, 'Geom2']
        
    # per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
    # all orifices in alpha are "bottom" 
    Cd = 0.65 # same for all 
    g = 32.2 # ft / s^2 (imperial units)
    # the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)
    
    while not done:
        
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state(level=level)
            
            # iterate over the action space entries containing "Or" (orifices)
            for idx in range(len(env.config['action_space'])):
                orifice_id = env.config['action_space'][idx]
                # the area of the orifice is the area of the circle
                Ao = orifice_areas[orifice_id]
                # Q = Cd * Ao * sqrt(2*g*H_e)
                Q_desired = optimal_constant_flows[idx] 

                # assume linear scaling with opening percentage
                if state[idx] > 0.0:
                    u_open_pct[idx] = Q_desired / (Cd * Ao * np.sqrt(2*g*state[idx]))
                else:
                    u_open_pct[idx] = 1.0
                
                if u_open_pct[idx] > 1.0:
                    u_open_pct[idx] = 1.0
                elif u_open_pct[idx] < 0.0:
                    u_open_pct[idx] = 0.0
            
            filling_degrees = state / max_depths_array
            filling_degree_avg = np.mean(filling_degrees)
            
            if evaluating == "constant-flow":

                done = env.step(u_open_pct.flatten())
            elif evaluating == "equal-filling":
                for idx in range(len(env.config['action_space'])):
                    this_fd = filling_degrees[idx]
                    u_open_pct[idx] += optimal_efd_params * (this_fd - filling_degree_avg)

                for i in range(len(u_open_pct)):
                    if u_open_pct[i] > 1.0:
                        u_open_pct[i] = 1.0
                    elif u_open_pct[i] < 0.0:
                        u_open_pct[i] = 0.0
                done = env.step(u_open_pct.flatten())

            else:
                print("error. control scenario not recongized.")
                done = True
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
            
            current_flows_dict = dict()
            for orifice in env.config['action_space']:
                current_flows_dict[orifice] = pyswmm.Links(env.env.sim)[orifice].flow
            current_flows = pd.DataFrame(data = current_flows_dict,columns=env.config['action_space'], index=[env.env.sim.current_time])
            
            flows = pd.concat((flows,current_flows))

                

        if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
            final_depths = env.state()

        #done = env.step(u_open_pct.flatten(),level=level)
        done = env.step(u_open_pct.flatten())
        
    perf = sum(env.data_log["performance_measure"])
    print("cost:")
    print("{:.4e}".format(perf))
    print("flooding")
    for key,value in env.data_log['flooding'].items():
        # flooded nodes
        if sum(value) > 0:
            print(key, sum(value))
    print("flow threshold exceedance")
    for key,value in env.data_log['flow'].items():
        value = [x-4.0 for x in value]
        value = [x if x > 0 else 0 for x in value]
        # high flows
        if sum(value) > 0:
            print(key, sum(value))
    print("nodes with ending depth > 0.1 ft")
    nodes_not_empty = 0
    for key,value in env.data_log['depthN'].items():
        if value[-1] > 0.10:
            print(key)
            nodes_not_empty += 1
    print("cost from nodes not empty: ", "{:.4e}".format(nodes_not_empty*(7*10**5)))
            
    # round final_depths to 2 decimal places
    final_depths = np.round(final_depths,2)
    # calcuate the final filling degrees of all measured depths
    final_filling_degrees = np.round(final_depths / max_depths_array , 2)
    # save the cost and ending filling degree to a csv
    perf_summary = pd.DataFrame(data = {"cost": perf, "final_depths": final_depths,"final_filling": final_filling_degrees})
    if evaluating == "uncontrolled":
        perf_summary.to_csv(str(folder_path + "/costs_" + evaluating + ".csv"))
    else:
        perf_summary.to_csv(str(folder_path + "/costs_" + evaluating + "_a=" + str(parameter) + ".csv"))

    if plot:
        
        
        # if there are any na values in states or weir_heads32, linearly interpolate over them
        states.interpolate(method='time',axis='index',inplace=True)
        #weir_heads32.interpolate(method='time',axis='index',inplace=True)
        actions.interpolate(method='time',axis='index',inplace=True)
        flows.interpolate(method='time',axis='index',inplace=True)

        plots_high = max(len(env.config['action_space']) , len(env.config['states']))
        fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

        axes[0,0].set_title("actions")
        axes[0,1].set_title("states")
        # plot the actions
        for idx in range(len(env.config['action_space'])):
            #axes[idx,0].plot(weir_heads32.iloc[:,idx])
            #axes[idx,0].set_ylabel("(weir head [ft])^(3/2)")
            axes[idx,0].plot(flows.iloc[:,idx])
            axes[idx,0].set_ylabel("flow")
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
            plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + "_param=" + str(parameter) + ".png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(evaluating) + "_param=" + str(parameter) + ".svg"),dpi=450)
        #plt.show()
        plt.close('all')