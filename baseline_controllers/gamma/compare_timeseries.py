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
level = "3"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.gamma(version=version,level=level)
env.env.sim.start()

# choose parameters for the plot
equal_filling_param = "0.0"
constant_flow_param = "0.0"

equal_filling_data_log = pd.read_pickle(str("./v" + version + "/lev" + level + "/results/equal-filling_param=" + str(equal_filling_param) + "_data_log.pkl"))
constant_flow_data_log = pd.read_pickle(str("./v" + version + "/lev" + level + "/results/constant-flow_param=" + str(constant_flow_param) + "_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/results/uncontrolled_data_log.pkl"))

# print the costs
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("equal filling: ", "{:.2E}".format(sum(equal_filling_data_log['performance_measure'])))
print("constant flow: ", "{:.2E}".format(sum(constant_flow_data_log['performance_measure'])))

# load the actions and states
uncontrolled_actions = pd.read_csv("./v" + version + "/results/actions_uncontrolled.csv",index_col=0,parse_dates=True)
uncontrolled_states = pd.read_csv("./v" + version + "/results/states_uncontrolled.csv",index_col=0,parse_dates=True)
equal_filling_actions = pd.read_csv(str("./v" + version + "/lev" + level + "/results/actions_equal-filling_param=" + equal_filling_param + ".csv"),index_col=0,parse_dates=True)
equal_filling_states = pd.read_csv(str("./v" + version + "/lev" + level + "/results/states_equal-filling_param=" + equal_filling_param + ".csv"),index_col=0,parse_dates=True)
constant_flow_actions = pd.read_csv(str("./v" + version + "/lev" + level + "/results/actions_constant-flow_param=" + constant_flow_param + ".csv"),index_col=0,parse_dates=True)
constant_flow_states = pd.read_csv(str("./v" + version + "/lev" + level + "/results/states_constant-flow_param=" + constant_flow_param + ".csv"),index_col=0,parse_dates=True)

plots_high = max(len(env.config['action_space']) , len(env.config['states']))

fig = plt.figure(figsize=(10,2*plots_high))
gs = GridSpec(plots_high,2,figure=fig)

# plot the actions
for idx, name in enumerate(env.config['action_space']):
    ax = fig.add_subplot(gs[idx,0] )  
    #ax.plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
    #ax.plot(equal_filling_actions.index, equal_filling_actions[env.config['action_space'][idx]], label='Equal Filling',color='blue',alpha=0.6)
    #ax.plot(constant_flow_actions.index, constant_flow_actions[env.config['action_space'][idx]], label='Constant Flow',color='red',alpha=0.6)
    ax.plot(uncontrolled_data_log['simulation_time'], uncontrolled_data_log['flow'][name], label = "Uncontrolled",color='black',alpha=0.6)
    ax.plot(equal_filling_data_log['simulation_time'], equal_filling_data_log['flow'][name], label = "Equal Filling",color='blue',alpha=0.6)
    ax.plot(constant_flow_data_log['simulation_time'], constant_flow_data_log['flow'][name], label = "Constant Flow",color='red',alpha=0.6)
    # horizontal dotted line at env.config._performormance_threshold
    ax.axhline(y=env._performormance_threshold, color='r', linestyle='--')


    if idx == len(env.config['action_space']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([equal_filling_actions.index[0],equal_filling_actions.index[int(len(equal_filling_actions.index)/2)],equal_filling_actions.index[-1]])
        
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
    ax.plot(equal_filling_states.index, equal_filling_states[str(env.config['states'][idx])], label='Equal Filling',color='blue',alpha=0.6)
    ax.plot(constant_flow_states.index, constant_flow_states[str(env.config['states'][idx])], label='Constant Flow',color='red',alpha=0.6)

    ax.annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    ax.axhline(y = pyswmm.Nodes(env.env.sim)[name[0]].full_depth, color='r', linestyle='--')


    if idx == len(env.config['states']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([equal_filling_states.index[0],equal_filling_states.index[int(len(equal_filling_states.index)/2)],equal_filling_states.index[-1]])
        
    if idx == 0:
        ax.set_title("Storage Depths")
    if idx != len(env.config['states']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])

    if idx == len(env.config['states']) -1 : # last row for the legend
        ax.legend(fontsize='x-large')

        
        

unc_perf = sum(uncontrolled_data_log['performance_measure'])
cf_perf = sum(constant_flow_data_log['performance_measure'])
ef_perf = sum(equal_filling_data_log['performance_measure'])

perfstr = "Cost Difference from Uncontrolled\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.55, 0.4), xycoords='axes fraction', ha='center', va='center',fontsize='x-large')
ax.axis('off')

plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".png")) 
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".svg"))
#plt.show()
plt.close('all')



# try arranging the main trunk in a U shape
import sys
sys.path.append("C:/modpods")
import modpods

flows = pd.DataFrame.from_dict(equal_filling_data_log['flow'])
depthN = pd.DataFrame.from_dict(equal_filling_data_log['depthN'])
depthN.columns = env.config['states']
flows.columns = env.config['action_space'] # to match the naming conventions on the subway map
equal_filling_response = pd.concat([flows, depthN], axis=1)
equal_filling_response.index = equal_filling_data_log['simulation_time']

flows = pd.DataFrame.from_dict(uncontrolled_data_log['flow'])
depthN = pd.DataFrame.from_dict(uncontrolled_data_log['depthN'])
depthN.columns = env.config['states']
flows.columns = env.config['action_space'] # to match the naming conventions on the subway map
uncontrolled_response = pd.concat([flows, depthN], axis=1)
uncontrolled_response.index = uncontrolled_data_log['simulation_time']

flows = pd.DataFrame.from_dict(constant_flow_data_log['flow'])
depthN = pd.DataFrame.from_dict(constant_flow_data_log['depthN'])
depthN.columns = env.config['states']
flows.columns = env.config['action_space'] # to match the naming conventions on the subway map
constant_flow_response = pd.concat([flows, depthN], axis=1)
constant_flow_response.index = constant_flow_data_log['simulation_time']

subway = modpods.subway_map_from_pystorms(env)

fig=plt.figure(figsize=(12,6))
ax = plt.subplot(111)
pos = nx.multipartite_layout(subway['graph'], subset_key='generation', align='vertical')

# arrange the main trunk in an L
# that main trunk is 9, 8, 6, 5, 4, 3, 2, 1
pos[('9','depthN')] = np.array([0.1,0.6])
pos['O9'] = np.array([0.1,0.5])
pos[('8','depthN')] = np.array([0.1,0.4])
pos['O8'] = np.array([0.1,0.3])
pos[('6','depthN')] = np.array([0.1,0.2])
pos['O6'] = np.array([0.1,0.1])
pos[('5','depthN')] = np.array([0.2,0.1])
pos['O5'] = np.array([0.3,0.1])
pos[('4','depthN')] = np.array([0.4,0.1])
pos['O4'] = np.array([0.5,0.1])
pos[('3','depthN')] = np.array([0.6,0.1])
pos['O3'] = np.array([0.6,0.2])
pos[('2','depthN')] = np.array([0.6,0.3])
pos['O2'] = np.array([0.6,0.4])
pos[('1','depthN')] = np.array([0.6,0.5])
pos['O1'] = np.array([0.6,0.6])
# 7 comes into 6, put it parallel with 8
pos[('7','depthN')] = np.array([0.2,0.4])
pos['O7'] = np.array([0.2,0.3])
# 10 goes into 4, put it parallel with 5
pos[('10','depthN')] = np.array([0.35,0.35])
pos['O10'] = np.array([0.35,0.25])
# 11 flows into 10
pos[('11','depthN')] = np.array([0.35,0.55])
pos['O11'] = np.array([0.35,0.45])

#nx.draw_networkx_nodes(subway['graph'], pos, node_size=500)
#nx.draw_networkx_labels(subway['graph'], pos, font_size=12)
nx.draw_networkx_edges(subway['graph'], pos, arrows=True,arrowsize=20,style='solid',alpha=0.3, min_source_margin = 20, min_target_margin= 400)
plt.tight_layout()

'''
# match each node with its timeseries
for node in subway['graph'].nodes():
    if node in response.columns:
        node['graph'] = node
'''
trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

graphsize = 2.3 / len(uncontrolled_response.columns)
g2 = graphsize / 2.0

depth_idx = 0
for n in subway['graph']:
    xx,yy=trans(pos[n]) # figure coordinates
    xa,ya=trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-g2,ya-g2, graphsize, graphsize])
    
    
    if "depth" in n[0] or "depth" in n[1]:
        #a.plot(modpods_response[n]/ft2meters,color='blue',alpha=0.6)
        a.plot(equal_filling_response[n],color='blue',alpha=0.6)
        a.plot(uncontrolled_response[n],color='black',alpha=0.6)
        a.plot(constant_flow_response[n],color='green',alpha=0.6)
        a.set_yticks([0, max(equal_filling_response[n]) ])
        depth_idx += 1
    else:
        #a.plot(modpods_response[n]/cfs2cms,color='blue',alpha=0.6)
        a.plot(equal_filling_response[n],color='blue',alpha=0.6)  
        a.plot(uncontrolled_response[n],color='black',alpha=0.6)
        a.plot(constant_flow_response[n],color='green',alpha=0.6)
        #a.plot(bayesian_response[n]/cfs2cms,color='green',alpha=0.6)
        #a.plot(response.index,np.ones(len(response.index))*0.11, color='red', linestyle='--')
        a.axhline(y= 4.0, color='red', linestyle='-')
        a.set_yticks([0, max(max(uncontrolled_response[n]) , 4.0 )])
    #a.axis('off')

    a.set_xticks([])
    
    # set the precision to one decimal place
    a.set_yticklabels([round(x,1) for x in a.get_yticks()])

    # make the tick marks invisible
    a.tick_params(axis='both', which='both',length=0)
        
    if isinstance(n,tuple):
        a.set_title(str("Basin " + n[0]),y=0.6)
        if "8" in n[0]:
            a.set_title(str("Basin " + n[0]),y=0.3)
    else:
        a.set_title(str("Valve " + n[1:]),y=0.6)
    # make none of the spines visible
    for spine in a.spines.values():
        spine.set_visible(False)
    # make the plot square
    a.set_aspect('auto',anchor='C',adjustable='box')

#modpods_patch = mpatches.Patch(color='blue', label='modpods',alpha=0.6)
equal_filling_patch = mpatches.Patch(color='blue', label='equal filling',alpha=0.6)
uncontrolled_patch = mpatches.Patch(color='black', label='uncontrolled',alpha=0.6)
constant_flow_patch = mpatches.Patch(color='green', label='constant flow',alpha=0.6)
#bayesian_patch = mpatches.Patch(color='green', label='bayesian',alpha=0.6)
ax.legend(handles=[equal_filling_patch,uncontrolled_patch,constant_flow_patch],
           loc=(0.1,0.7), fontsize='xx-large')

'''
# annotate the relative costs, lower center, xx-large
ax.annotate("Cost - Uncontrolled Cost:", xy=(0.5, 0.35), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.annotate("modpods: " + "{:.2E}".format(sum(modpods_data_log['performance_measure'])-sum(uncontrolled_data_log['performance_measure'])), 
            xy=(0.5, 0.3), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.annotate("equal filling: " + "{:.2E}".format(sum(equal_filling_data_log['performance_measure'])-sum(uncontrolled_data_log['performance_measure'])),
            xy=(0.5, 0.25), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.annotate("bayesian: " + "{:.2E}".format(sum(bayesian_data_log['performance_measure'])-sum(uncontrolled_data_log['performance_measure'])),
            xy=(0.5, 0.2), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

ax.annotate("Pystorms Scenario Gamma", xy=(0.5, 0.65), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
'''
    
ax.axis('off')
plt.savefig(str("./v" + version + "/lev" + level + "/evaluate_all_U_lev" + level + ".png"),dpi=450)
plt.savefig(str("./v" + version + "/lev" + level + "/evaluate_all_U_lev" + level + ".svg"),dpi=450)
plt.tight_layout()
#plt.show(block=True)
plt.close('all')


