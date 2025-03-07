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

# ALPHA SCENARIO
version = "2"
control = "constant-flow" # 'equal-filling' and 'constant-flow' - not 'structural' as it has no real-time control
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.alpha(version=version)
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

# don't plot the states with "C" in them
states_to_plot = [state for state in env.config['states'] if "C" not in state[0]]

plots_high = max(len(env.config['action_space']) , len(states_to_plot))

fig = plt.figure(figsize=(10,2*plots_high))
gs = GridSpec(plots_high,2,figure=fig)

# plot the actions
for idx in range(len(env.config['action_space'])):
    ax = fig.add_subplot(gs[idx,0] )  
    #ax.plot(uncontrolled_actions.index, uncontrolled_actions[env.config['action_space'][idx]], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(level1_actions.index, level1_actions[env.config['action_space'][idx]], label='Level 1',color='blue',alpha=0.6)
    ax.plot(level2_actions.index, level2_actions[env.config['action_space'][idx]], label='Level 2',color='green',alpha=0.6)
    ax.plot(level3_actions.index, level3_actions[env.config['action_space'][idx]], label='Level 3',color='red',alpha=0.6)

    
    if idx == len(env.config['action_space']) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([level1_actions.index[0],level1_actions.index[int(len(level1_actions.index)/2)],level1_actions.index[-1]])
        
    if idx == 0:
        ax.set_title("Actions")
    if idx != len(env.config['action_space']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the states
for idx in range(len(states_to_plot)):
    ax = fig.add_subplot(gs[idx,1] )  
    ax.plot(uncontrolled_states.index, uncontrolled_states[str(states_to_plot[idx])], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(level1_states.index, level1_states[str(states_to_plot[idx])], label='Level 1',color='blue',alpha=0.6)
    ax.plot(level2_states.index, level2_states[str(states_to_plot[idx])], label='Level 2',color='green',alpha=0.6)
    ax.plot(level3_states.index, level3_states[str(states_to_plot[idx])], label='Level 3',color='red',alpha=0.6)

    ax.annotate(str(states_to_plot[idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
    


    if idx == len(states_to_plot) - 1:
        ax.set_xlabel("time")
        # just add ticks in the beginning, middle, and end of the index
        ax.set_xticks([level1_states.index[0],level1_states.index[int(len(level1_states.index)/2)],level1_states.index[-1]])
        
    if idx == 0:
        ax.set_title("States")
    if idx != len(states_to_plot) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])

    if idx == len(states_to_plot) - 2: # second to last row, for the legend
        ax = fig.add_subplot(gs[idx,0])
        ax.plot(uncontrolled_states.index[0:2], np.zeros((2,1)), label = 'Uncontrolled',color='black',alpha=0.6)
        ax.plot(level1_states.index[0:2], np.zeros((2,1)), label = 'Level 1',color='blue',alpha=0.6)
        ax.plot(level2_states.index[0:2], np.zeros((2,1)), label = 'Level 2',color='green',alpha=0.6)
        ax.plot(level3_states.index[0:2], np.zeros((2,1)), label = 'Level 3',color='red',alpha=0.6)
        ax.legend(fontsize='x-large')
        

unc_perf = sum(uncontrolled_data_log['performance_measure'])
level1_perf = sum(level1_data_log['performance_measure'])
level2_perf = sum(level2_data_log['performance_measure'])
level3_perf = sum(level3_data_log['performance_measure'])

#perfstr = "Cost Difference from Uncontrolled\nStructual = {:+.1%}\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((struct_perf - unc_perf)/unc_perf,(cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
perfstr = "Cost Difference from Uncontrolled\nLevel 1 = {:+.1%}\nLevel 2 = {:+.1%}\nLevel 3 = {:+.1%}".format((level1_perf - unc_perf)/unc_perf,(level2_perf - unc_perf)/unc_perf, (level3_perf - unc_perf)/unc_perf)

ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.axis('off')

plt.tight_layout()
# only going to use one plot for level (at most) so don't worry about tracking parameters
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.svg"))
#plt.show()
plt.close('all')

# plot system layout of graphs

#response = pd.concat([actions, states],axis=1)
unc_response = pd.concat([uncontrolled_actions, uncontrolled_states],axis=1)
level1_response = pd.concat([level1_actions, level1_states],axis=1)
level2_response = pd.concat([level2_actions, level2_states],axis=1)
level3_response = pd.concat([level3_actions, level3_states],axis=1)



index = list(list(env.config['states']) + list(env.config['action_space']))  
# remove duplicates in index
index = list(set(index))
adjacency = pd.DataFrame(columns = index, index = index)
# define the adjacency connections so we can visualize flow
# this just has "1's" to represent flow from the row to the column
# so this will be inverted from connectivity and read as: "to (column)" - "from (row)" - "1 for flow", "0 for no flow"
# flows to the WWTP (JIout):
adjacency[('R1','depthN')][('J1','depthN')] = 1
#adjacency[('I5','flow')][('R1','depthN')] = 1
adjacency[('R2','depthN')][('J2','depthN')] = 1
#adjacency[('I5','flow')][('R2','depthN')] = 1
adjacency[('R3','depthN')][('J3','depthN')] = 1
#adjacency[('I5','flow')][('R3','depthN')] = 1
adjacency[('R4','depthN')][('J4','depthN')] = 1
#adjacency[('I5','flow')][('R4','depthN')] = 1
adjacency[('R5','depthN')][('J5a','depthN')] = 1
#adjacency[('I5','flow')][('R5','depthN')] = 1
# CSO's into the river:
#adjacency['W1'][('R1','depthN')] = 1
adjacency[('JC1b','depthN')][('R1','depthN')] = 1
#adjacency[('C5b','depthL')]['W1'] = 1
#adjacency['W2'][('R2','depthN')] = 1
adjacency[('JC2','depthN')][('R2','depthN')] = 1
#adjacency[('C5b','depthL')]['W2'] = 1
#adjacency['W3'][('R3','depthN')] = 1
adjacency[('JC3a','depthN')][('R3','depthN')] = 1
#adjacency[('C5b','depthL')]['W3'] = 1

adjacency[('JC4c','depthN')][('R4','depthN')] = 1

#adjacency['W4'][('R4','depthN')] = 1
#adjacency[('C5b','depthL')]['W4'] = 1
#adjacency['W5'][('R5','depthN')] = 1
adjacency[('JC5','depthN')][('R5','depthN')] = 1
#adjacency[('C5b','depthL')]['W5'] = 1
# fill the na's in adjacency with "0" for "no connection"
adjacency.fillna(0,inplace=True)


# drop any rows or columns which are only zeros
adjacency = adjacency.loc[(adjacency!=0).any(axis=1), (adjacency!=0).any(axis=0)]
# reindex the adjacency matrix to reflect the dropped rows and columns
index = list(adjacency.columns)
index.extend(list(adjacency.index))
index = list(set(index))
adjacency = adjacency.reindex(index=index,columns=index)

# make the adjacency matrix square, adding entries of 0 if necessary
for col in index:
    if col not in adjacency.columns:
        adjacency[col] = 0
for row in index:
    if row not in adjacency.index:
        adjacency.loc[row] = 0


#print the shape of adjacency to make sure it's square
adjacency = adjacency[adjacency.index]
# fill na's with 0
adjacency.fillna(0,inplace=True)
#print(adjacency.shape)
#print(adjacency)


graph = nx.from_pandas_adjacency(adjacency,create_using=nx.DiGraph)
for node in graph.nodes:
    if "JC" in node[0] or "JC" in node[1]:
        graph.nodes[node]['level'] = 1
    elif "J" in node[0] or "J" in node[1]:
        graph.nodes[node]['level'] = 3
    elif "R" in node[0] or "R" in node[1]:
        graph.nodes[node]['level'] = 2
    else:
        graph.nodes[node]['level'] = 0



subway = {'adjacency':adjacency,'index':index,'graph':graph}



# plot the flows on top of the subway map
fig=plt.figure(figsize=(16,8))
ax = plt.subplot(111)
pos = nx.multipartite_layout(subway['graph'],subset_key = 'level',align='horizontal')
for node in pos:
    if "1" in node[0] or "1" in node[1]:
        pos[node][0] = -1
    elif "2" in node[0] or "2" in node[1]:
        pos[node][0] = -0.5
    elif "3" in node[0] or "3" in node[1]:
        pos[node][0] = 0.0
    elif "4" in node[0] or "4" in node[1]:
        pos[node][0] = 0.5
    elif "5" in node[0] or "5" in node[1]:
        #if "C" not in node[0] and "C" not in node[1] and "I" not in node[0] and "I" not in node[1]:
        pos[node][0] = 1.0


#nx.draw_networkx_nodes(subway['graph'], pos, node_size=500)
#nx.draw_networkx_labels(subway['graph'], pos, font_size=12)
nx.draw_networkx_edges(subway['graph'], pos, arrows=True,arrowsize=20,style='solid',alpha=0.5, min_source_margin = 50, min_target_margin=70)
#plt.tight_layout()


trans = ax.transData.transform
trans2 = fig.transFigure.inverted().transform

graphsize = 3.0 / len(unc_response.columns)
g2 = graphsize / 2.0

max_depths = dict()
for state in env.config['states']:
    if 'depthN' in state[1]:
        node_id = state[0]
        max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth

for n in subway['graph']:
    xx,yy=trans(pos[n]) # figure coordinates
    xa,ya=trans2((xx,yy)) # axes coordinates
    a = plt.axes([xa-g2,ya-g2, graphsize, graphsize])
    
    
    if "depthN" in n[0] or "depthN" in n[1]:
        a.plot(unc_response[str(n)], label ='Uncontrolled',color='black',alpha=0.6)
        a.plot(level1_response[str(n)], label ='Level 1',color='blue',alpha=0.6)
        a.plot(level2_response[str(n)], label ='Level 2',color='green',alpha=0.6)
        a.plot(level3_response[str(n)], label ='Level 3',color='red',alpha=0.6)
        
        a.plot(unc_response.index,np.ones(len(unc_response.index))*max_depths[n[0]], color='red', linestyle='--')
        #a.plot(response[n])
        #a.plot(response.index,np.ones(len(response.index))*max_depths[n[0]], color='red', linestyle='--')
        #a.axhline(y=max_depths[n[0]] , color='r', linestyle='-') # leave the threshold off for characterization
        a.set_yticks([0,  max_depths[n[0]] ])
    else:
        #a.plot(response[n])    
        a.plot(unc_response[str(n)], label ='Uncontrolled',color='black',alpha=0.6)
        a.plot(level1_response[str(n)], label ='Level 1',color='blue',alpha=0.6)
        a.plot(level2_response[str(n)], label ='Level 2',color='green',alpha=0.6)
        a.plot(level3_response[str(n)], label ='Level 3',color='red',alpha=0.6)
        
        #a.plot(response.index,np.ones(len(response.index))*0.11, color='red', linestyle='--')
        #a.axhline(y= 3.9 , color='r', linestyle='-')
        a.set_yticks([0, 1 ])
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

# make the legend
unc_patch = mpatches.Patch(color='black', label='Uncontrolled')
level1_patch = mpatches.Patch(color='blue', label='Level 1')
level2_patch = mpatches.Patch(color='green', label='Level 2')
level3_patch = mpatches.Patch(color='red', label='Level 3')
# put the legend slightly below the center of the figure
#plt.legend(handles=[unc_patch,cf_patch,ef_patch,struct_patch], loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)
plt.legend(handles=[unc_patch,level1_patch,level2_patch,level3_patch], loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=3)


ax.axis('off')
plt.savefig(str("./v" + version + "/" + control + "_compare_levels_subway.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels_subway.svg"))

#plt.tight_layout()
#plt.show()
plt.close('all')