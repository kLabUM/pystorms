import pystorms
import pyswmm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import datetime
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os


# GAMMA SCENARIO
version = "2"
control = "equal-filling" # "equal-filling" or "constant-flow"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.gamma(version=version)
env.env.sim.start()

# choose parameters for the plot
equal_filling_param = "0.0"
constant_flow_param = "0.0"

level1_data_log = pd.read_pickle(str("./v" + version + "/lev1/results/" + control + "_param=0.0_data_log.pkl"))
level2_data_log = pd.read_pickle(str("./v" + version + "/lev2/results/" + control + "_param=0.0_data_log.pkl"))
level3_data_log = pd.read_pickle(str("./v" + version + "/lev3/results/" + control + "_param=0.0_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/results/uncontrolled_data_log.pkl"))
# print the costs
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("level 1: ", "{:.2E}".format(sum(level1_data_log['performance_measure'])))
print("level 2: ", "{:.2E}".format(sum(level2_data_log['performance_measure'])))
print("level 3: ", "{:.2E}".format(sum(level3_data_log['performance_measure'])))


# load the actions and states
uncontrolled_actions = pd.read_csv("./v" + version + "/results/actions_uncontrolled.csv",index_col=0,parse_dates=True)
uncontrolled_states = pd.read_csv("./v" + version + "/results/states_uncontrolled.csv",index_col=0,parse_dates=True)
level1_actions = pd.read_csv("./v" + version + "/lev1/results/actions_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)
level1_states = pd.read_csv("./v" + version + "/lev1/results/states_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)
level2_actions = pd.read_csv("./v" + version + "/lev2/results/actions_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)
level2_states = pd.read_csv("./v" + version + "/lev2/results/states_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)
level3_actions = pd.read_csv("./v" + version + "/lev3/results/actions_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)
level3_states = pd.read_csv("./v" + version + "/lev3/results/states_" + control + "_param=0.0.csv",index_col=0,parse_dates=True)



plots_high = max(len(env.config['action_space']) , len(env.config['states']))

fig = plt.figure(figsize=(10,2*plots_high))
gs = GridSpec(plots_high,2,figure=fig)

# plot the actions
for idx, name in enumerate(env.config['action_space']):
    ax = fig.add_subplot(gs[idx,0] )  
    ax.plot(uncontrolled_data_log['simulation_time'], uncontrolled_data_log['flow'][name], label = "Uncontrolled",color='black',alpha=0.6)
    ax.plot(level1_data_log['simulation_time'], level1_data_log['flow'][name], label = "Level 1",color='blue',alpha=0.6)
    ax.plot(level2_data_log['simulation_time'], level2_data_log['flow'][name], label = "Level 2",color='green',alpha=0.6)
    ax.plot(level3_data_log['simulation_time'], level3_data_log['flow'][name], label = "Level 3",color='red',alpha=0.6)

    # horizontal dotted line at env.config._performormance_threshold
    ax.axhline(y=env._performormance_threshold, color='r', linestyle='--')


    if idx == len(env.config['action_space']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([uncontrolled_data_log['simulation_time'][0],uncontrolled_data_log['simulation_time'][int(len(uncontrolled_data_log['simulation_time'])/2)],uncontrolled_data_log['simulation_time'][-1]])
        
    if idx == 0:
        ax.set_title("Valve Flows")
    if idx != len(env.config['action_space']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])
        
    ax.annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the states
for idx, name in enumerate(env.config['states']):
    ax = fig.add_subplot(gs[idx,1] )  
    ax.plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(level1_states.index, level1_states[str(env.config['states'][idx])], label='Level 1',color='blue',alpha=0.6)
    ax.plot(level2_states.index, level2_states[str(env.config['states'][idx])], label='Level 2',color='green',alpha=0.6)
    ax.plot(level3_states.index, level3_states[str(env.config['states'][idx])], label='Level 3',color='red',alpha=0.6)
    
    ax.annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    ax.axhline(y = pyswmm.Nodes(env.env.sim)[name[0]].full_depth, color='r', linestyle='--')


    if idx == len(env.config['states']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([uncontrolled_data_log['simulation_time'][0],uncontrolled_data_log['simulation_time'][int(len(uncontrolled_data_log['simulation_time'])/2)],uncontrolled_data_log['simulation_time'][-1]])
        
    if idx == 0:
        ax.set_title("Storage Depths")
    if idx != len(env.config['states']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])

    if idx == len(env.config['states']) -1 : # last row for the legend
        ax.legend(fontsize='x-large')

        
        

unc_perf = sum(uncontrolled_data_log['performance_measure'])
lev1_perf = sum(level1_data_log['performance_measure'])
lev2_perf = sum(level2_data_log['performance_measure'])
lev3_perf = sum(level3_data_log['performance_measure'])

#perfstr = "Cost Difference from Uncontrolled\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
perfstr = "Cost Difference from Uncontrolled\nLevel 1 = {:+.1%}\nLevel 2 = {:+.1%}\nLevel 3 = {:+.1%}".format((lev1_perf - unc_perf)/unc_perf, (lev2_perf - unc_perf)/unc_perf, (lev3_perf - unc_perf)/unc_perf)

ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.55, 0.4), xycoords='axes fraction', ha='center', va='center',fontsize='x-large')
ax.axis('off')

plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.svg"))
#plt.show()
plt.close('all')



