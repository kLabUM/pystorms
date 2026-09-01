from cProfile import label
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

# DELTA SCENARIO
version = "2"
control = "prop-outflow" # "static-plus-rule" or "prop-outflow"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.delta(version=version)
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
    
plots_high = max(len(env.config['action_space']) , len(env.config['states']))
fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

axes[0,0].set_title("actions")
axes[0,1].set_title("states")
# plot the actions
for idx in range(len(env.config['action_space'])):
    axes[idx,0].plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
    axes[idx,0].plot(level1_actions.index, level1_actions[env.config['action_space'][idx]], label='Level 1',color='blue',alpha=0.6)
    axes[idx,0].plot(level2_actions.index, level2_actions[env.config['action_space'][idx]], label='Level 2',color='green',alpha=0.6)
    axes[idx,0].plot(level3_actions.index, level3_actions[env.config['action_space'][idx]], label='Level 3',color='red',alpha=0.6)
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
#axes[-1,0].plot(env.data_log['flow']['conduit_Eout'])
axes[-1,0].plot(uncontrolled_data_log['simulation_time'],uncontrolled_data_log['flow']['conduit_Eout'],label="uncontrolled",color='black',alpha=0.6)
axes[-1,0].plot(level1_data_log['simulation_time'],level1_data_log['flow']['conduit_Eout'],label="level 1",color='blue',alpha=0.6)
axes[-1,0].plot(level2_data_log['simulation_time'],level2_data_log['flow']['conduit_Eout'],label="level 2",color='green',alpha=0.6)
axes[-1,0].plot(level3_data_log['simulation_time'],level3_data_log['flow']['conduit_Eout'],label="level 3",color='red',alpha=0.6)
axes[-1,0].set_ylabel("flow")
axes[-1,0].set_xlabel("time")
# add a dotted red line at threshold
axes[-1,0].axhline(y=env.threshold, color='r', linestyle='dotted')


# plot the states
for idx, state in enumerate(env.config['states']):
    axes[idx,1].plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    axes[idx,1].plot(level1_states.index, level1_states[str(env.config['states'][idx])], label='Level 1',color='blue',alpha=0.6)
    axes[idx,1].plot(level2_states.index, level2_states[str(env.config['states'][idx])], label='Level 2',color='green',alpha=0.6)
    axes[idx,1].plot(level3_states.index, level3_states[str(env.config['states'][idx])], label='Level 3',color='red',alpha=0.6)

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

    if idx == 2:
        # put the legend in middle of axes
        axes[idx,1].legend()
    
    if idx != len(env.config['states']) - 1:
        axes[idx,1].set_xticklabels([])
        axes[idx,1].annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')


unc_perf = sum(uncontrolled_data_log['performance_measure'])
lev1_perf = sum(level1_data_log['performance_measure'])
lev2_perf = sum(level2_data_log['performance_measure'])
lev3_perf = sum(level3_data_log['performance_measure'])

#perfstr = "Cost Difference from Uncontrolled\nStatic+Rule = {:+.1%}\nProp-Outflow = {:+.1%}".format((static_perf - unc_perf)/unc_perf, (prop_perf - unc_perf)/unc_perf)
perfstr = "Cost Difference from Uncontrolled\nLevel 1 = {:+.1%}\nLevel 2 = {:+.1%}\nLevel 3 = {:+.1%}".format((lev1_perf - unc_perf)/unc_perf, (lev2_perf - unc_perf)/unc_perf, (lev3_perf - unc_perf)/unc_perf)

axes[-2,1].annotate(perfstr, xy=(0.5, 0.45), xycoords='axes fraction', ha='center', va='center',fontsize='large')


plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.svg"))
#plt.show()
plt.close('all')