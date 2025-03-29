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
level = "1"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.delta(version=version,level=level)
env.env.sim.start()


# choose parameters for the plot
static_plus_rule_param = "0.0"
prop_outflow_param = "0.0"

static_plus_rule_data_log = pd.read_pickle(str("./v" + version + "/lev" + level + "/results/static-plus-rule_param=" + str(static_plus_rule_param) + "_data_log.pkl"))
prop_outflow_data_log = pd.read_pickle(str("./v" + version + "/lev" + level + "/results/prop-outflow_param=" + str(prop_outflow_param) + "_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/results/uncontrolled_data_log.pkl"))

# print the costs
print("Costs")
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("Static+Rule: ", "{:.2E}".format(sum(static_plus_rule_data_log['performance_measure'])))
print("Prop Outflow: ", "{:.2E}".format(sum(prop_outflow_data_log['performance_measure'])))

# load the actions and states
uncontrolled_actions = pd.read_csv("./v" + version + "/results/actions_uncontrolled.csv",index_col=0,parse_dates=True)
uncontrolled_states = pd.read_csv("./v" + version + "/results/states_uncontrolled.csv",index_col=0,parse_dates=True)
static_plus_rule_actions = pd.read_csv(str("./v" + version + "/lev" + level + "/results/actions_static-plus-rule_param=" + static_plus_rule_param + ".csv"),index_col=0,parse_dates=True)
static_plus_rule_states = pd.read_csv(str("./v" + version + "/lev" + level + "/results/states_static-plus-rule_param=" + static_plus_rule_param + ".csv"),index_col=0,parse_dates=True)
prop_outflow_actions = pd.read_csv(str("./v" + version + "/lev" + level + "/results/actions_prop-outflow_param=" + prop_outflow_param + ".csv"),index_col=0,parse_dates=True)
prop_outflow_states = pd.read_csv(str("./v" + version + "/lev" + level + "/results/states_prop-outflow_param=" + prop_outflow_param + ".csv"),index_col=0,parse_dates=True)

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
    

# record total outflows under each control scenario
static_plus_rule_total_outflows = sum(static_plus_rule_data_log['flow']['conduit_Eout'])
prop_outflow_total_outflows = sum(prop_outflow_data_log['flow']['conduit_Eout'])
uncontrolled_total_outflows = sum(uncontrolled_data_log['flow']['conduit_Eout'])
# print those total outflows
print("Total outflows from system through Eout")
print("Static+Rule: ", "{:.2E}".format(static_plus_rule_total_outflows))
print("Prop Outflow: ", "{:.2E}".format(prop_outflow_total_outflows))
print("Uncontrolled: ", "{:.2E}".format(uncontrolled_total_outflows))

# print the final depths in each storage asset under each control scenario
print("final depths in storage basins")
print("Static+Rule: \n", static_plus_rule_states.iloc[-1,:])
print("Prop Outflow: \n", prop_outflow_states.iloc[-1,:])
print("Uncontrolled: \n", uncontrolled_states.iloc[-1,:])

# print the last date
print("last timestep")
print("static+rule: ", static_plus_rule_data_log['simulation_time'][-1])
print("prop outflow: ", prop_outflow_data_log['simulation_time'][-1])
print("uncontrolled: ", uncontrolled_data_log['simulation_time'][-1])
print("number of timesteps")
print("static+rule: ", len(static_plus_rule_data_log['simulation_time']))
print("prop outflow: ", len(prop_outflow_data_log['simulation_time']))
print("uncontrolled: ", len(uncontrolled_data_log['simulation_time']))

plots_high = max(len(env.config['action_space']) , len(env.config['states']))
fig, axes = plt.subplots(plots_high, 2, figsize=(10,2*plots_high))

axes[0,0].set_title("actions")
axes[0,1].set_title("states")
# plot the actions
for idx in range(len(env.config['action_space'])):
    #axes[idx,0].plot(weir_heads32.iloc[:,idx])
    #axes[idx,0].set_ylabel("(weir head [ft])^(3/2)")
    axes[idx,0].plot(static_plus_rule_actions.index, static_plus_rule_actions[env.config['action_space'][idx]], label='Static+Rule',color='blue',alpha=0.6)
    axes[idx,0].plot(prop_outflow_actions.index, prop_outflow_actions[env.config['action_space'][idx]], label='Prop Outflow',color='red',alpha=0.6)
    axes[idx,0].plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
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
axes[-1,0].plot(static_plus_rule_data_log['simulation_time'],static_plus_rule_data_log['flow']['conduit_Eout'],label="static+rule",color='blue',alpha=0.6)
axes[-1,0].plot(prop_outflow_data_log['simulation_time'],prop_outflow_data_log['flow']['conduit_Eout'],label="prop outflow",color='red',alpha=0.6)
axes[-1,0].set_ylabel("flow")
axes[-1,0].set_xlabel("time")
# add a dotted red line at threshold
axes[-1,0].axhline(y=env.threshold, color='r', linestyle='dotted')


# plot the states
for idx, state in enumerate(env.config['states']):
    axes[idx,1].plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    axes[idx,1].plot(static_plus_rule_states.index, static_plus_rule_states[str(env.config['states'][idx])], label='Static+Rule',color='blue',alpha=0.6)
    axes[idx,1].plot(prop_outflow_states.index, prop_outflow_states[str(env.config['states'][idx])], label='Prop Outflow',color='red',alpha=0.6)

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
prop_perf = sum(prop_outflow_data_log['performance_measure'])
static_perf = sum(static_plus_rule_data_log['performance_measure'])

perfstr = "Cost Difference from Uncontrolled\nStatic+Rule = {:+.1%}\nProp-Outflow = {:+.1%}".format((static_perf - unc_perf)/unc_perf, (prop_perf - unc_perf)/unc_perf)
axes[-2,1].annotate(perfstr, xy=(0.5, 0.45), xycoords='axes fraction', ha='center', va='center',fontsize='large')


plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".png")) 
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".svg"))
#plt.show()
plt.close('all')