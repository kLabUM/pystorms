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

np.set_printoptions(precision=3,suppress=True)

# EPSILON
# options are: 'equal-filling' and 'constant-flow' (or 'uncontrolled')
control_scenario = 'equal-filling' 
verbose = True
version = "2" # options are "1" and "2"
level = "1" # options are "1" , "2", and "3"
plot = False # plot True significantly increases the memory usage. 
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
# set the random seed
rand_seed = 42
np.random.seed(rand_seed)


print("evaluating ", control_scenario, " for epsilon scenario")

tuning_values = np.arange(-0.05,0.25,0.01)
tuning_values = np.round(tuning_values,2)

# for dev or plotting - single value
#tuning_values = [0.0]

if control_scenario == "constant-flow" or control_scenario == "equal-filling":
    folder_path = str("./v" + version + "/lev" + level + "/results")
elif control_scenario == "uncontrolled":
    folder_path = str("./v" + version + "/results")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for parameter in tuning_values:
    optimal_constant_flows = np.loadtxt(str("./v" + version + "/optimal_constant_flows.txt"))
    optimal_efd_params = np.loadtxt(str("./v" + version + "/optimal_efd_params.txt"))

    print("tuning value: ", parameter)
    optimal_constant_flows = optimal_constant_flows*(1+parameter)

    # project file is in english units
    cfs2cms = 35.315
    ft2meters = 3.281
    env = pystorms.scenarios.epsilon(version=version,level=level)

    env.env.sim.start()
    done = False
    
    u_open_pct = np.ones((len(env.config['action_space']),1)) # begin all open

    h32 = -1.0*np.ones((len(env.config['action_space']),1)) # weir head ^ (3/2)

    last_eval = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts
    last_read = env.env.sim.start_time - datetime.timedelta(days=1) # initto a time before the simulation starts

    states = pd.DataFrame(columns = env.config['states'])
    weir_heads32 = pd.DataFrame(columns = env.config['action_space'])
    actions = pd.DataFrame(columns = env.config['action_space'])

    print(env.env.actuator_schedule)
    print(env.env.sensor_schedule)

    # per https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf the equation for flow over a transverse weir 
    # (all control assets in epsilon are transverse weirs) is:
    # Q = Cw L h^(1.5)
    # where Q is the flow, Cw is the weir discharge coefficient, L is the length of the weir, and h is the head over the weir
    # modpods will pick up the constants in each case. so we just need to provide the weir head to the correct power
    # the maximum hiehgt of the weir varies by asset
    H = {"ISD001": 14.7, "ISD002": 9.0, "ISD003": 14.0, "ISD004": 15.5, "ISD005": 15.5, "ISD006": 15.5, "ISD007": 15.5, "ISD008": 12.25, "ISD009": 15.5, "ISD010": 10.5, "ISD011": 11.5}
    H_array = [14.7,9.0,14.0, 15.5 , 15.5 ,15.5, 15.5 ,12.25,15.5,10.5 ,11.5]
    max_depths_array = np.array([])
    max_depths = dict()
    for state in env.config['states']:
        if 'depth' in state[1]:
            node_id = state[0]
            max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)
            
    while not done:
        
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            state = env.state(level=level)
            #print("clean," , env.state(level="1"))
            #print("level 2," , env.state(level="2"))
            #print("level 3," , env.state(level="3"))
            if control_scenario == 'equal-filling' or control_scenario == 'constant-flow':
            
                for idx in range(len(u_open_pct)): # set opening percentage to achieve the desired flow rate
                    desired_head = optimal_constant_flows[idx]
                    max_weir_height = H[env.config['action_space'][idx]]
                    
                    h_upstream = state[idx]
                    if desired_head > h_upstream: # all the way open
                        u_open_pct[idx,0] = 1.0
                    else:
                        h_weir = h_upstream - desired_head
                        closed_percentage = h_weir / max_weir_height
                        u_open_pct[idx,0] = 1.0 - closed_percentage
                if control_scenario == "equal-filling":
                    u_avg = np.mean(u_open_pct)
                    u_diff = u_avg - u_open_pct
                    for i in [0,3,8]: # most downstream controllers
                        TSS_conc = env.state()[-1]
                        delta_TSS = TSS_conc - 150 # 150 is long term average
                        u_diff[i,0] = optimal_efd_params[0]*delta_TSS # downstream respond to TSS, not others filling degrees
                    u_open_pct = u_open_pct + optimal_efd_params[1]*u_diff
                
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0] > 1.0:
                        u_open_pct[i,0] = 1.0
                    elif u_open_pct[i,0] < 0.09:
                        u_open_pct[i,0] = 0.09
            elif control_scenario == 'uncontrolled':
                u_open_pct = np.ones((len(env.config['action_space']),1))
            if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0: 
                u_print = u_open_pct.flatten()
                y_measured = env.state().reshape(-1,1)
                print("              y_measured,  u")
                print(np.c_[np.array(env.config['states'][:11]),np.round(y_measured[:11],2) , np.round(u_print.reshape(-1,1),3)])
                print("current time, end time")
                print(env.env.sim.current_time, env.env.sim.end_time)
                print("\n")
            
        if (env.env.sim.current_time > last_read + datetime.timedelta(minutes=1)) and plot: # log data 
            last_read = env.env.sim.current_time
            state = env.state().reshape(1,len(env.config['states']))
            current_state = pd.DataFrame(data=state, columns = env.config['states'], index = [env.env.sim.current_time] )
            states = pd.concat((states,current_state))
            action = u_open_pct.reshape(1,len(env.config['action_space']))
            current_actions = pd.DataFrame(data = action, columns = env.config['action_space'], index=[env.env.sim.current_time])
            actions = pd.concat((actions, current_actions))
    
            weir_heads32_rn = np.array([])
            for asset_idx in range(len(env.config['action_space'])):
                #print("junction ", env.config['states'][asset_idx], " is upstream of ", env.config['action_space'][asset_idx])
                h_up = env.state()[asset_idx]
                h_weir = (1 - u_open_pct[asset_idx])*H[env.config['action_space'][asset_idx]] # weir height
                if h_weir >= h_up: # weir is taller than the water pooled behind it, so no flow
                    h = 0
                elif h_up > h_weir: # assume weir is free flowing (negligble backwater effects)
                    h = h_up - h_weir
            
                weir_heads32_rn = np.append(weir_heads32_rn, h**(3/2))
        
            weir_heads32_rn = weir_heads32_rn.reshape(1,len(env.config['action_space']))
            weir_heads32 = pd.concat((weir_heads32,pd.DataFrame(data=weir_heads32_rn, columns=env.config['action_space'], index = [env.env.sim.current_time])),axis=0)
            
        if env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
            final_depths = env.state()[:11]

        done = env.step(u_open_pct.flatten(),level=level)
    
    perf = sum(env.data_log["performance_measure"])
    print("cost:")
    print("{:.4e}".format(perf))
    print("cost from flooding (by asset id)")
    flood_cost = 0
    for key,value in env.data_log['flooding'].items():
        if sum(value) > 0:
            # the total cost is the number of positive entries in value multiplied by the cost of flooding (10e6)
            # find the number of positive entries in value
            num_positive = sum([1 for x in value if x > 0])
            print(key,"{:.4e}".format(num_positive*(10**9)))
            flood_cost += num_positive*(10**9)

    print("cost ignoring flooding")
    print("{:.4e}".format(perf - flood_cost))
    
    # calculate final filling degrees based on the max_depths array
    final_filling_degrees = np.array([depth/max_depth for depth,max_depth in zip(final_depths,max_depths_array)])
    
    # save the cost and ending filling degree to a csv
    perf_summary = pd.DataFrame(data = {"cost": perf, "final_depths": final_depths,"final_filling": final_filling_degrees})
    if control_scenario == "uncontrolled":
        perf_summary.to_csv(str(folder_path + "/costs_" + control_scenario + ".csv"))
    else:
        perf_summary.to_csv(str(folder_path + "/costs_" + control_scenario + "_a=" + str(parameter) + ".csv"))


    if plot:
        
        
        # if there are any na values in states or weir_heads32, linearly interpolate over them
        states.interpolate(method='time',axis='index',inplace=True)
        weir_heads32.interpolate(method='time',axis='index',inplace=True)
        actions.interpolate(method='time',axis='index',inplace=True)

        plots_high = max(len(env.config['action_space']) , len(env.config['states']))
        fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

        axes[0,0].set_title("weirs")
        axes[0,1].set_title("junctions")
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

        # plot the states
        for idx in range(len(env.config['states'])):
            axes[idx,1].plot(states.iloc[:,idx])
            if "L" in env.config['states'][idx][1]:
                pass
            else:
                axes[idx,1].axhline(y=max_depths_array[idx],color='r')
                if idx <= 10: # controlled juinctions 
                    axes[idx,1].axhline(y=H_array[idx],color='k')
                axes[idx,1].set_ylabel("depth (ft)")

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
        if control_scenario == "uncontrolled":
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + ".png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + ".svg"),dpi=450)
        else:
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_param=" + str(parameter) + ".png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_param=" + str(parameter) + ".svg"),dpi=450)
        #plt.show()
        plt.close('all')

        # put all the data together
        #response = pd.concat([weir_heads32, states], axis=1)
        response = pd.concat([actions, states],axis=1)
        import sys
        sys.path.append("C:/modpods")
        import modpods
        import networkx as nx
        # plot the flows on top of the subway map
        subway = modpods.subway_map_from_pystorms(env)

        fig=plt.figure(figsize=(16,8))
        ax = plt.subplot(111)
        pos = nx.multipartite_layout(subway['graph'], subset_key='generation', align='vertical')
        # move 44, ISD009 004, and ISD001 all a bit to the right so there's less arrow overlap

        pos[('004','depthN')][0] = pos[('004','depthN')][0] + 0.5
        pos['ISD001'][0] = pos['ISD001'][0] + 0.5
        pos[('044','depthN')][0] = pos[('044','depthN')][0] + 0.5
        pos['ISD009'][0] = pos['ISD009'][0] + 0.5

        #nx.draw_networkx_nodes(subway['graph'], pos, node_size=500)
        #nx.draw_networkx_labels(subway['graph'], pos, font_size=12)
        nx.draw_networkx_edges(subway['graph'], pos, arrows=True,arrowsize=20,style='solid',alpha=0.5, min_source_margin = 50, min_target_margin=50)
        plt.tight_layout()

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform

        graphsize = 1.1 / len(response.columns)
        g2 = graphsize / 2.0

        depth_idx = 0
        for n in subway['graph']:
            xx,yy=trans(pos[n]) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            a = plt.axes([xa-g2,ya-g2, graphsize, graphsize])
    
    
            if "depth" in n[0] or "depth" in n[1]:
                a.plot(response[n])
                #a.plot(response.index,np.ones(len(response.index))*basin_max_depths[depth_idx]/ft2meters, color='red', linestyle='--')
                #a.axhline(y=max_depths[n[0]] , color='r', linestyle='-') # leave the threshold off for characterization
                #a.set_yticks([0, max(max(response[n]/ft2meters) , max_depths[n[0]] )])
                depth_idx += 1
            else:
                a.plot(response[n])    
                #a.plot(response.index,np.ones(len(response.index))*0.11, color='red', linestyle='--')
                #a.axhline(y= 3.9 , color='r', linestyle='-')
                #a.set_yticks([0, max(max(response[n]) , 1 )])
            #a.axis('off')

            a.set_xticks([])
    
            # set the precision to one decimal place
            a.set_yticklabels([round(x,1) for x in a.get_yticks()])

            # make the tick marks invisible
            a.tick_params(axis='both', which='both',length=0)
        
            if isinstance(n,tuple):
                a.set_title(n[0])
            else:
                a.set_title(n)
            # make none of the spines visible
            for spine in a.spines.values():
                spine.set_visible(False)
            # make the plot square
            a.set_aspect('auto',anchor='C',adjustable='box')

    
        ax.axis('off')
        if control_scenario == "uncontrolled":
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_subway.png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_subway.svg"),dpi=450)
        else:
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_param=" + str(parameter) + "_subway.png"),dpi=450)
            plt.savefig(str(folder_path + "/evaluate_" + str(control_scenario) + "_param=" + str(parameter) + "_subway.svg"),dpi=450)
        plt.tight_layout()
        #plt.show()
        plt.close('all')

        # resample the flows and depths to 5 minute intervals
        weir_heads32 = weir_heads32.resample('5min').mean()
        states = states.resample('5min').mean()
        actions = actions.resample('5min').mean()

        if control_scenario == "uncontrolled":
            weir_heads32.to_csv(str(folder_path + "/weir_heads32_" + str(control_scenario) + ".csv"))
            states.to_csv(str(folder_path + "/states_" + str(control_scenario) + ".csv"))
            actions.to_csv(str(folder_path + "/actions_" + str(control_scenario) + ".csv"))
            # and the data log
            with open(f'./v{version}/results/{control_scenario}_data_log.pkl', 'wb') as f:
                pickle.dump(env.data_log, f)
        else:
            # save the flows and depths
            weir_heads32.to_csv(str(folder_path + "/weir_heads32_" + str(control_scenario) + "_param=" + str(parameter) + ".csv"))
            states.to_csv(str(folder_path + "/states_" + str(control_scenario) + "_param=" + str(parameter) + ".csv"))
            actions.to_csv(str(folder_path + "/actions_" + str(control_scenario) + "_param=" + str(parameter) + ".csv"))
            # and the data log
            with open(f'./v{version}/lev{level}/results/{str(control_scenario + "_param=" + str(parameter))}_data_log.pkl', 'wb') as f:
                pickle.dump(env.data_log, f)