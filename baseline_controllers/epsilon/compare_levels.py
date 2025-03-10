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
control = "equal-filling" # "equal-filling" or "constant-flow"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.epsilon(version=version)
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
        ax.set_title("Weirs [Fraction Open]")
    if idx != len(env.config['action_space']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the states
for idx in range(len(env.config['states'])):
    ax = fig.add_subplot(gs[idx,1] )  
    ax.plot(uncontrolled_states.index, uncontrolled_states[str(env.config['states'][idx])], label='Uncontrolled',color='black',alpha=0.6)
    ax.plot(level1_states.index, level1_states[str(env.config['states'][idx])], label='Level 1',color='blue',alpha=0.6)
    ax.plot(level2_states.index, level2_states[str(env.config['states'][idx])], label='Level 2',color='green',alpha=0.6)
    ax.plot(level3_states.index, level3_states[str(env.config['states'][idx])], label='Level 3',color='red',alpha=0.6)
    

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
        ax.set_xticks([level1_actions.index[0],level1_actions.index[int(len(level1_actions.index)/2)],level1_actions.index[-1]])
        
    if idx == 0:
        ax.set_title("States")
    if idx != len(env.config['states']) - 1: # not the last row
        ax.set_xticks([])
        ax.set_xticklabels([])

    if idx == len(env.config['states']) - 2: # second to last row, for the legend
        ax = fig.add_subplot(gs[idx,0])
        ax.plot(uncontrolled_states.index[0:2], np.zeros((2,1)), label = 'Uncontrolled',color='black',alpha=0.6)
        ax.plot(level1_states.index[0:2], np.zeros((2,1)), label = 'Level 1',color='blue',alpha=0.6)
        ax.plot(level2_states.index[0:2], np.zeros((2,1)), label = 'Level 2',color='green',alpha=0.6)
        ax.plot(level3_states.index[0:2], np.zeros((2,1)), label = 'Level 3',color='red',alpha=0.6)
        ax.axis('off')
        ax.legend(fontsize='x-large')
        

unc_perf = sum(uncontrolled_data_log['performance_measure'])
level1_perf = sum(level1_data_log['performance_measure'])
level2_perf = sum(level2_data_log['performance_measure'])
level3_perf = sum(level3_data_log['performance_measure'])

#perfstr = "Cost Difference from Uncontrolled\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
perfstr = "Cost Difference from Uncontrolled\nLevel 1 = {:+.1%}\nLevel 2 = {:+.1%}\nLevel 3 = {:+.1%}".format((level1_perf - unc_perf)/unc_perf, (level2_perf - unc_perf)/unc_perf, (level3_perf - unc_perf)/unc_perf)

ax = fig.add_subplot(gs[-1,0])
ax.annotate(perfstr, xy=(0.5, 0.6), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')
ax.axis('off')

plt.tight_layout()
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels.svg"))
#plt.show()
plt.close('all')

################
# plot the timeseries in an L
################
import sys
sys.path.append("C:/modpods")
import modpods

uncontrolled_response = pd.concat([uncontrolled_actions,uncontrolled_states],axis=1)
level1_response = pd.concat([level1_actions,level1_states],axis=1)
level2_response = pd.concat([level2_actions,level2_states],axis=1)
level3_response = pd.concat([level3_actions,level3_states],axis=1)

# put the plots in an L
subway = modpods.subway_map_from_pystorms(env)

# cut off the first twelve hours of the responses ( 5 minute resolution )
uncontrolled_response = uncontrolled_response.iloc[12*12:]
level1_response = level1_response.iloc[12*12:]
level2_response = level2_response.iloc[12*12:]
level3_response = level3_response.iloc[12*12:]



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
        a.plot(level1_response[str(n)],color='blue',alpha=0.6)
        a.plot(level2_response[str(n)],color='green',alpha=0.6)
        a.plot(level3_response[str(n)],color='red',alpha=0.6)
        
        #a.plot(response.index,np.ones(len(response.index))*basin_max_depths[depth_idx]/ft2meters, color='red', linestyle='--')
        if depth_idx < 11: # none of the junctions come close to flooding, so not worth showing the threshold
            #a.axhline(y=max_depths[n[0]] , color='red', linestyle='-',alpha=0.5) # leave the threshold off for characterization
            #a.set_yticks([0, max_depths[n[0]] ])
            #a.set_yticks([0, max(max(equal_filling_response[str(n)]) , max(constant_flow_response[str(n)]) )])
            a.set_yticks([0, max(max(level1_response[str(n)]) , max(level2_response[str(n)]) , max(level3_response[str(n)]) , max(uncontrolled_response[str(n)]) )])
            
        # for the outlet make the ticks at the min and max
        if n[0] == "001" and n[1] == "depthN":
            a.set_yticks([min(uncontrolled_response[str(n)]) , max(uncontrolled_response[str(n)])])
        depth_idx += 1
    elif "L" in n[1]:
        a.plot(uncontrolled_response["('001', 'pollutantL', 'TSS')"],color='black',alpha=0.6)
        a.plot(level1_response["('001', 'pollutantL', 'TSS')"],color='blue',alpha=0.6)
        a.plot(level2_response["('001', 'pollutantL', 'TSS')"],color='green',alpha=0.6)
        a.plot(level3_response["('001', 'pollutantL', 'TSS')"],color='red',alpha=0.6)

        
    else:
        #a.plot(uncontrolled_response[n],color='black',alpha=0.6)
        a.plot(level1_response[n],color='blue',alpha=0.6)
        a.plot(level2_response[n],color='green',alpha=0.6)
        a.plot(level3_response[n],color='red',alpha=0.6)
       
        
        #a.plot(response.index,np.ones(len(response.index))*0.11, color='red', linestyle='--')
        #a.axhline(y= 3.9 , color='r', linestyle='-')
        #a.set_yticks([min(min(equal_filling_response[n]), min(constant_flow_response[n])) , min(1.0,max(max(equal_filling_response[n]), max(constant_flow_response[n])))  ])
        a.set_yticks([min(min(level1_response[n]), min(level2_response[n]), min(level3_response[n]), min(uncontrolled_response[n])) , min(1.0,max(max(level1_response[n]), max(level2_response[n]), max(level3_response[n]), max(uncontrolled_response[n])))  ])
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
lev1_patch = mpatches.Patch(color='blue', label='level 1',alpha=0.6)
lev2_patch = mpatches.Patch(color='green', label='level 2',alpha=0.6)
lev3_patch = mpatches.Patch(color='red', label='level 3',alpha=0.6)

ax.legend(handles=[uncontrolled_patch,lev1_patch,lev2_patch,lev3_patch],
           loc=(0.8,0.05), fontsize='xx-large')

# annotate the relative costs
ax.annotate(perfstr, xy=(0.9, 0.9), xycoords='axes fraction', ha='center', va='center',fontsize='large')

ax.axis('off')
plt.savefig(str("./v" + version + "/" + control + "_compare_levels_L.png")) 
plt.savefig(str("./v" + version + "/" + control + "_compare_levels_L.svg"))
#plt.show()
plt.close('all')