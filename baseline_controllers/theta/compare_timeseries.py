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


# THETA SCENARIO
version = "1"
level = "1"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.theta(version=version,level=level)
env.env.sim.start()

p1_max_depth = pyswmm.Nodes(env.env.sim)["P1"].full_depth
p2_max_depth = pyswmm.Nodes(env.env.sim)["P2"].full_depth
flow_threshold = env.threshold

equal_filling_data_log = pd.read_pickle(str("./v" + version + "/equal-filling_level_" + str(level) + "_data_log.pkl"))
constant_flow_data_log = pd.read_pickle(str("./v" + version + "/constant-flow_level_" + str(level) + "_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/uncontrolled_level_" + str(level) + "_data_log.pkl"))

# print the costs
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("equal filling: ", "{:.2E}".format(sum(equal_filling_data_log['performance_measure'])))
print("constant flow: ", "{:.2E}".format(sum(constant_flow_data_log['performance_measure'])))

# load the actions and states
uncontrolled_flows = pd.read_csv(str("./v" + version + "/flows_uncontrolled_level_" + str(level) + ".csv"))
uncontrolled_depths = pd.read_csv(str("./v" + version + "/depths_uncontrolled_level_" + str(level) + ".csv"))
equal_filling_flows = pd.read_csv(str("./v" + version + "/flows_equal-filling_level_" + str(level) + ".csv"))
equal_filling_depths = pd.read_csv(str("./v" + version + "/depths_equal-filling_level_" + str(level) + ".csv"))
constant_flow_flows = pd.read_csv(str("./v" + version + "/flows_constant-flow_level_" + str(level) + ".csv"))
constant_flow_depths = pd.read_csv(str("./v" + version + "/depths_constant-flow_level_" + str(level) + ".csv"))

fig = plt.figure(figsize=(10,10))
gs = GridSpec(3, 2, figure=fig)

# plot the depths on the upper two plots
ax_1depth = fig.add_subplot(gs[0, 0])
ax_2depth = fig.add_subplot(gs[0, 1])
ax_1flow = fig.add_subplot(gs[1, 0])
ax_2flows = fig.add_subplot(gs[1, 1])
ax_8flows = fig.add_subplot(gs[2, :])

linewidth = 3

ax_1depth.plot(equal_filling_depths.index, equal_filling_depths["('P1', 'depthN')"], label='equal filling', color='blue', alpha=0.6,linewidth=linewidth)
ax_1depth.plot(uncontrolled_depths.index, uncontrolled_depths["('P1', 'depthN')"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_1depth.plot(constant_flow_depths.index, constant_flow_depths["('P1', 'depthN')"], label='constant flow', color='green', alpha=0.6,linewidth=linewidth)
# horizontal line at max depth, dotted red
ax_1depth.axhline(p1_max_depth, color='red', linestyle='--', label='Threshold',linewidth=linewidth)
ax_1depth.set_ylabel('$m$',rotation=0, fontsize='xx-large', labelpad = 15)
# remove the x ticks
ax_1depth.set_xticks([])
# set the title
ax_1depth.set_title('Depth in P1',y=0.8,fontsize='xx-large')
# add a legend to this axis, top right x-large
ax_1depth.legend(loc='right', fontsize='x-large')

# plot the depth in P2
ax_2depth.plot(equal_filling_depths.index, equal_filling_depths["('P2', 'depthN')"], label='equal filling', color='blue', alpha=0.6,linewidth=linewidth)
ax_2depth.plot(uncontrolled_depths.index, uncontrolled_depths["('P2', 'depthN')"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_2depth.plot(constant_flow_depths.index, constant_flow_depths["('P2', 'depthN')"], label='constant flow', color='green', alpha=0.6,linewidth=linewidth)
# horizontal line at max depth, dotted red
ax_2depth.axhline(p2_max_depth, color='red', linestyle='--', label='Threshold',linewidth=linewidth)
#ax_2depth.set_ylabel('$m$',rotation=0, fontsize='xx-large', labelpad = 15)
# remove the x ticks
ax_2depth.set_xticks([])
# set the title
ax_2depth.set_title('Depth in P2',y=0.8,fontsize='xx-large')

# plot flows out of p1
ax_1flow.plot(equal_filling_flows.index, equal_filling_flows["1"], label='equal filling', color='blue', alpha=0.6,linewidth=linewidth)
ax_1flow.plot(uncontrolled_flows.index, uncontrolled_flows["1"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_1flow.plot(constant_flow_flows.index, constant_flow_flows["1"], label='constant flow', color='green', alpha=0.6,linewidth=linewidth)
ax_1flow.set_ylabel('$m^3 / s$',rotation=0, fontsize='xx-large', labelpad = 25)
# title
ax_1flow.set_title('Flow out of P1',y=0.8,fontsize='xx-large')
ax_1flow.set_xticks([])

unc_perf = sum(uncontrolled_data_log['performance_measure'])
cf_perf = sum(constant_flow_data_log['performance_measure'])
ef_perf = sum(equal_filling_data_log['performance_measure'])

perfstr = "Cost Difference from Uncontrolled\nConstant Flow = {:+.1%}\nEqual Filling = {:+.1%}".format((cf_perf - unc_perf)/unc_perf, (ef_perf - unc_perf)/unc_perf)
ax_8flows.annotate(perfstr, xy=(0.7, 0.5), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')


# plot flows out of p2
ax_2flows.plot(equal_filling_flows.index, equal_filling_flows["2"], label='equal filling', color='blue', alpha=0.6,linewidth=linewidth)
ax_2flows.plot(uncontrolled_flows.index, uncontrolled_flows["2"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_2flows.plot(constant_flow_flows.index, constant_flow_flows["2"], label='constant flow', color='green', alpha=0.6,linewidth=linewidth)
#ax_2flows.set_ylabel('$m^3 / s$',rotation=0, fontsize='xx-large', labelpad = 25)
# title
ax_2flows.set_title('Flow out of P2',y=0.8,fontsize='xx-large')
# remove the x ticks
ax_2flows.set_xticks([])

# plot flows at 8 (confluence)
ax_8flows.plot(equal_filling_data_log['simulation_time'], equal_filling_data_log['flow']['8'], label='equal filling', color='blue', alpha=0.6,linewidth=linewidth)
ax_8flows.plot(uncontrolled_data_log['simulation_time'], uncontrolled_data_log['flow']['8'], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_8flows.plot(constant_flow_data_log['simulation_time'], constant_flow_data_log['flow']['8'], label='constant flow', color='green', alpha=0.6,linewidth=linewidth)
ax_8flows.set_ylabel('$m^3 / s$',rotation=0, fontsize='xx-large', labelpad = 25)
# threshold line
ax_8flows.axhline(flow_threshold, color='red', linestyle='--', label='Threshold',linewidth=linewidth)
# title
ax_8flows.set_title('Flow at 8',y=0.8,fontsize='xx-large')


# make the ticklabels bigger
ax_8flows.tick_params(axis='both', labelsize='x-large')
ax_1depth.tick_params(axis='both', labelsize='x-large')
ax_2depth.tick_params(axis='both', labelsize='x-large')
ax_1flow.tick_params(axis='both', labelsize='x-large')
ax_2flows.tick_params(axis='both', labelsize='x-large')

#fig.title("Scenario Theta | Version " + version + " | Level " + level, fontsize=20)
# add title to the figure
fig.suptitle("Scenario Theta | Version " + version + " | Level " + level, fontsize=20)

plt.tight_layout()
plt.savefig("v" + version + "/level_" + str(level) + "_compare_timeseries.png", dpi=300)
plt.savefig("v" + version + "/level_" + str(level) + "_compare_timeseries.svg", dpi=300)
plt.show()
