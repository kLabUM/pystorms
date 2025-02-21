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

# BETA SCENARIO
version = "1"
level = "1"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
#env = pystorms.scenarios.beta(version=version,level=level)
env = pystorms.scenarios.beta()
env.env.sim.start()


mpc_data_log = pd.read_pickle(str("./v" + version + "/lev" + level + "/results/mpc_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/results/uncontrolled_data_log.pkl"))

# print the costs
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("mpc: ", "{:.2E}".format(sum(mpc_data_log['performance_measure'])))

# load the actions and states
uncontrolled_actions = pd.read_csv("./v" + version + "/results/actions_uncontrolled.csv",index_col=0,parse_dates=True)
uncontrolled_states = pd.read_csv("./v" + version + "/results/states_uncontrolled.csv",index_col=0,parse_dates=True)
mpc_actions = pd.read_csv("./v" + version + "/lev" + level + "/results/actions_mpc.csv",index_col=0,parse_dates=True)
mpc_states = pd.read_csv("./v" + version + "/lev" + level + "/results/states_mpc.csv",index_col=0,parse_dates=True)


plots_high = max(len(env.config['action_space']) , len(env.config['states']))

fig = plt.figure(figsize=(10,2*plots_high))
gs = GridSpec(plots_high,2,figure=fig)

# plot the actions
for idx, name in enumerate(env.config['action_space']):
    ax = fig.add_subplot(gs[idx,0] )  
    ax.plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(mpc_actions.index, mpc_actions[env.config['action_space'][idx]], label='MPC',color='green',alpha=0.6)

    if idx == len(env.config['action_space']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([mpc_actions.index[0],mpc_actions.index[int(len(mpc_actions.index)/2)],mpc_actions.index[-1]])
        
    if idx == 0:
        ax.set_title("Controls")
    if idx != len(env.config['action_space']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])
        
    ax.annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

ax = fig.add_subplot(gs[idx+1,0])        
ax.legend(fontsize='x-large')   

# plot the states
for idx, name in enumerate(env.config['states']):
    ax = fig.add_subplot(gs[idx,1] )  
    ax.plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(mpc_states.index, mpc_states[str(env.config['states'][idx])], label='MPC',color='green',alpha=0.6)

    ax.annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    ax.axhline(y = pyswmm.Nodes(env.env.sim)[name[0]].full_depth, color='r', linestyle='--')


    if idx == len(env.config['states']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([mpc_actions.index[0],mpc_actions.index[int(len(mpc_actions.index)/2)],mpc_actions.index[-1]])
        
    if idx == 0:
        ax.set_title("States")
    if idx != len(env.config['states']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])


unc_perf = sum(uncontrolled_data_log['performance_measure'])
mpc_perf = sum(mpc_data_log['performance_measure'])
perfstr = "Cost Difference from Uncontrolled\nModel Predictive Control = {:+.1%}".format((mpc_perf - unc_perf)/unc_perf)
ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.55, 0.4), xycoords='axes fraction', ha='center', va='center',fontsize='x-large')
ax.axis('off')

plt.tight_layout()
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states.png")) 
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states.svg"))
#plt.show()
plt.close('all')
