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

# EPSILON SCENARIO
version = "2"
level = "3"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.epsilon(version=version,level=level)
env.env.sim.start()

H = {"ISD001": 14.7, "ISD002": 9.0, "ISD003": 14.0, "ISD004": 15.5, "ISD005": 15.5, "ISD006": 15.5, "ISD007": 15.5, "ISD008": 12.25, "ISD009": 15.5, "ISD010": 10.5, "ISD011": 11.5}
H_array = [14.7,9.0,14.0, 15.5 , 15.5 ,15.5, 15.5 ,12.25,15.5,10.5 ,11.5]
max_depths_array = np.array([])
max_depths = dict()
for state in env.config['states']:
    if 'depth' in state[1]:
        node_id = state[0]
        max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
        max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)
        
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
for idx in range(len(env.config['action_space'])):
    ax = fig.add_subplot(gs[idx,0] )  
    #ax.plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(equal_filling_actions.index, equal_filling_actions[env.config['action_space'][idx]], label='Equal Filling',color='blue',alpha=0.6)
    ax.plot(constant_flow_actions.index, constant_flow_actions[env.config['action_space'][idx]], label='Constant Flow',color='red',alpha=0.6)
    
    if idx == len(env.config['action_space']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([equal_filling_actions.index[0],equal_filling_actions.index[int(len(equal_filling_actions.index)/2)],equal_filling_actions.index[-1]])
        
    if idx == 0:
        ax.set_title("Weirs [Fraction Open]")
    if idx != len(env.config['action_space']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the states
for idx in range(len(env.config['states'])):
    ax = fig.add_subplot(gs[idx,1] )  
    ax.plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(equal_filling_states.index, equal_filling_states[str(env.config['states'][idx])], label='Equal Filling',color='blue',alpha=0.6)
    ax.plot(constant_flow_states.index, constant_flow_states[str(env.config['states'][idx])], label='Constant Flow',color='red',alpha=0.6)
    
    if "L" in env.config['states'][idx][1]:
        ax.set_ylabel("TSS")
        ax.annotate(str(env.config['states'][idx]), xy=(0.6, 0.2), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    else:
        if idx <= 10: # controlled junctions 
            ax.axhline(y=max_depths_array[idx],color='red',alpha=0.5)
            ax.axhline(y=H_array[idx],color='k',alpha=0.3)
        ax.set_ylabel("depth (ft)")
        ax.annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    


    if idx == len(env.config['states']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([equal_filling_states.index[0],equal_filling_states.index[int(len(equal_filling_states.index)/2)],equal_filling_states.index[-1]])
        
    if idx == 0:
        ax.set_title("States")
    if idx != len(env.config['states']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])

    if idx == len(env.config['states']) - 2: # second to last row, for the legend
        ax = fig.add_subplot(gs[idx,0])
        ax.plot(uncontrolled_states.index[0:2], np.zeros((2,1)), label = 'Uncontrolled',color='black',alpha=0.6)
        ax.plot(equal_filling_states.index[0:2], np.zeros((2,1)), label = 'Equal Filling',color='blue',alpha=0.6)
        ax.plot(constant_flow_states.index[0:2], np.zeros((2,1)), label = 'Constant Flow',color='red',alpha=0.6)
        ax.axis('off')
        ax.legend(fontsize='x-large')
        

unc_perf = sum(uncontrolled_data_log['performance_measure'])
cf_perf = sum(constant_flow_data_log['performance_measure'])
ef_perf = sum(equal_filling_data_log['performance_measure'])

perfstr = "Cost Difference from Uncontrolled\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.axis('off')

plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".png")) 
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_lev" + level + ".svg"))
#plt.show()
plt.close('all')

################
# plot the timeseries in an L
################
import sys
sys.path.append("C:/modpods")
import modpods

uncontrolled_response = pd.concat([uncontrolled_actions,uncontrolled_states],axis=1)
constant_flow_response = pd.concat([constant_flow_actions,constant_flow_states],axis=1)
equal_filling_response = pd.concat([equal_filling_actions,equal_filling_states],axis=1)

# put the plots in an L
subway = modpods.subway_map_from_pystorms(env)

# cut off the first twelve hours of the responses ( 5 minute resolution )
uncontrolled_response = uncontrolled_response.iloc[12*12:]
constant_flow_response = constant_flow_response.iloc[12*12:]
equal_filling_response = equal_filling_response.iloc[12*12:]

# remove 001, polluantL, TSS from the graph
subway['graph'].remove_node(("001", "pollutantL", "TSS"))

fig=plt.figure(figsize=(12,6))
ax = plt.subplot(111)
pos = nx.multipartite_layout(subway['graph'], subset_key='generation', align='vertical')
# the main trunk is 39-isd008-33-isd007-30-isd006-27-isd005-22-isd004-001

pos[('039','depthN')] = np.array([0.1,0.6])
pos['ISD008'] = np.array([0.1,0.5])
pos[('033','depthN')] = np.array([0.1,0.4])
pos['ISD007'] = np.array([0.1,0.3])
pos[('030','depthN')] = np.array([0.1,0.2])
pos['ISD006'] = np.array([0.1,0.1])
pos[('027','depthN')] = np.array([0.2,0.1])
pos['ISD005'] = np.array([0.3,0.1])
pos[('022','depthN')] = np.array([0.4,0.1])
pos['ISD004'] = np.array([0.5,0.1])
pos[('001','depthN')] = np.array([0.75,0.35]) # this one will be bigger than the others
# won't plot the pollutant concentration, just the depth

# the 4, 6, 11 branch
pos[('006','depthN')] = np.array([0.25,0.6])
pos['ISD002'] = np.array([0.35,0.6])
pos[('011','depthN')] = np.array([0.25,0.5])
pos['ISD003'] = np.array([0.35,0.5])
pos[('004','depthN')] = np.array([0.45,0.55])
pos['ISD001'] = np.array([0.55,0.55])

# the 44, 50, 60 branch
pos[('050','depthN')] = np.array([0.25,0.35])
pos['ISD010'] = np.array([0.35,0.35])
pos[('060','depthN')] = np.array([0.25,0.25])
pos['ISD011'] = np.array([0.35,0.25])
pos[('044','depthN')] = np.array([0.45,0.3])
pos['ISD009'] = np.array([0.55,0.3])


nx.draw_networkx_edges(subway['graph'], pos, arrows=False,arrowsize=20,style='solid',alpha=0.3, min_source_margin = 200, min_target_margin= 200)


trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

graphsize = 1.6 / len(uncontrolled_response.columns)
g2 = graphsize / 2.0

depth_idx = 0

for n in subway['graph']:
    if n[0] == "001" and n[1] == "depthN":
        graphsize = graphsize*4 # outlet
        g2 = graphsize / 2.0
    xx,yy=trans(pos[n]) # figure coordinates
    xa,ya=trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-g2,ya-g2, graphsize, graphsize])
    if n[0] == "001" and n[1] == "depthN":
        graphsize = graphsize/4 # return to normal after making axis
        g2 = graphsize / 2.0
    
    
    if "depth" in n[0] or "depth" in n[1]:
        a.plot(uncontrolled_response[str(n)],color='black',alpha=0.6)
        a.plot(constant_flow_response[str(n)],color='red',alpha=0.6)
        a.plot(equal_filling_response[str(n)],color='blue',alpha=0.6)
        #a.plot(response.index,np.ones(len(response.index))*basin_max_depths[depth_idx]/ft2meters, color='red', linestyle='--')
        if depth_idx < 11: # none of the junctions come close to flooding, so not worth showing the threshold
            #a.axhline(y=max_depths[n[0]] , color='red', linestyle='-',alpha=0.5) # leave the threshold off for characterization
            #a.set_yticks([0, max_depths[n[0]] ])
            a.set_yticks([0, max(max(equal_filling_response[str(n)]) , max(constant_flow_response[str(n)]) )])
        # for the outlet make the ticks at the min and max
        if n[0] == "001" and n[1] == "depthN":
            a.set_yticks([min(uncontrolled_response[str(n)]) , max(uncontrolled_response[str(n)])])
        depth_idx += 1
    elif "L" in n[1]:
        a.plot(uncontrolled_response["('001', 'pollutantL', 'TSS')"],color='black',alpha=0.6)
        a.plot(constant_flow_response["('001', 'pollutantL', 'TSS')"],color='red',alpha=0.6)
        a.plot(equal_filling_response["('001', 'pollutantL', 'TSS')"],color='blue',alpha=0.6)
        
    else:
        #a.plot(uncontrolled_response[n],color='black',alpha=0.6)
        a.plot(constant_flow_response[n],color='red',alpha=0.6)
        a.plot(equal_filling_response[n],color='blue',alpha=0.6)
        
        #a.plot(response.index,np.ones(len(response.index))*0.11, color='red', linestyle='--')
        #a.axhline(y= 3.9 , color='r', linestyle='-')
        a.set_yticks([min(min(equal_filling_response[n]), min(constant_flow_response[n])) , min(1.0,max(max(equal_filling_response[n]), max(constant_flow_response[n])))  ])
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
    
uncontrolled_patch = mpatches.Patch(color='black', label='uncontrolled',alpha=0.6)
cf_patch = mpatches.Patch(color='red', label='constant flow',alpha=0.6)
ef_patch = mpatches.Patch(color='blue', label='equal filling',alpha=0.6)

ax.legend(handles=[uncontrolled_patch,cf_patch,ef_patch],
           loc=(0.8,0.05), fontsize='xx-large')

# annotate the relative costs
ax.annotate(perfstr, xy=(0.9, 0.9), xycoords='axes fraction', ha='center', va='center',fontsize='large')

ax.axis('off')
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_L_lev" + level + ".png")) 
plt.savefig(str("./v" + version + "/lev" + level + "/actions_states_L_lev" + level + ".svg"))
#plt.show()
plt.close('all')