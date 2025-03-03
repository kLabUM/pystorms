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
version = "2"
control = "equal-filling" # or "constant-flow"
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
env = pystorms.scenarios.theta(version=version)
env.env.sim.start()


p1_max_depth = pyswmm.Nodes(env.env.sim)["P1"].full_depth
p2_max_depth = pyswmm.Nodes(env.env.sim)["P2"].full_depth
flow_threshold = env.threshold

level1_data_log = pd.read_pickle(str("./v" + version + "/" + control + "_level_1_data_log.pkl"))
level2_data_log = pd.read_pickle(str("./v" + version + "/" + control + "_level_2_data_log.pkl"))
level3_data_log = pd.read_pickle(str("./v" + version + "/" + control + "_level_3_data_log.pkl"))
uncontrolled_data_log = pd.read_pickle(str("./v" + version + "/uncontrolled_level_1_data_log.pkl"))


# print the costs
print("uncontrolled: ", "{:.2E}".format(sum(uncontrolled_data_log['performance_measure'])))
print("level 1: ", "{:.2E}".format(sum(level1_data_log['performance_measure'])))
print("level 2: ", "{:.2E}".format(sum(level2_data_log['performance_measure'])))
print("level 3: ", "{:.2E}".format(sum(level3_data_log['performance_measure'])))

# load the actions and states
uncontrolled_flows = pd.read_csv(str("./v" + version + "/flows_uncontrolled_level_1.csv"))
uncontrolled_depths = pd.read_csv(str("./v" + version + "/depths_uncontrolled_level_1.csv"))
level1_flows = pd.read_csv(str("./v" + version + "/flows_" + control + "_level_1.csv"))
level1_depths = pd.read_csv(str("./v" + version + "/depths_" + control + "_level_1.csv"))
level2_flows = pd.read_csv(str("./v" + version + "/flows_" + control + "_level_2.csv"))
level2_depths = pd.read_csv(str("./v" + version + "/depths_" + control + "_level_2.csv"))
level3_flows = pd.read_csv(str("./v" + version + "/flows_" + control + "_level_3.csv"))
level3_depths = pd.read_csv(str("./v" + version + "/depths_" + control + "_level_3.csv"))

fig = plt.figure(figsize=(10,10))
gs = GridSpec(3, 2, figure=fig)

# plot the depths on the upper two plots
ax_1depth = fig.add_subplot(gs[0, 0])
ax_2depth = fig.add_subplot(gs[0, 1])
ax_1flow = fig.add_subplot(gs[1, 0])
ax_2flows = fig.add_subplot(gs[1, 1])
ax_8flows = fig.add_subplot(gs[2, :])

linewidth = 3

ax_1depth.plot(level1_depths.index, level1_depths["('P1', 'depthN')"], label='level 1', color='blue', alpha=0.6,linewidth=linewidth)
ax_1depth.plot(level2_depths.index, level2_depths["('P1', 'depthN')"], label='level 2', color='green', alpha=0.6,linewidth=linewidth)
ax_1depth.plot(level3_depths.index, level3_depths["('P1', 'depthN')"], label='level 3', color='red', alpha=0.6,linewidth=linewidth)
ax_1depth.plot(uncontrolled_depths.index, uncontrolled_depths["('P1', 'depthN')"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
# horizontal line at max depth, dotted red
ax_1depth.axhline(p1_max_depth, color='red', linestyle='--', label='Threshold',linewidth=linewidth)
ax_1depth.set_ylabel('$m$',rotation=0, fontsize='xx-large', labelpad = 15)
# remove the x ticks
ax_1depth.set_xticks([])
# add a legend
ax_1depth.legend(loc='right', fontsize='x-large')
ax_1depth.set_title('Depth in P1',y=0.8,fontsize='xx-large')

# plot the depth in P2
ax_2depth.plot(level1_depths.index, level1_depths["('P2', 'depthN')"], label='level 1', color='blue', alpha=0.6,linewidth=linewidth)
ax_2depth.plot(level2_depths.index, level2_depths["('P2', 'depthN')"], label='level 2', color='green', alpha=0.6,linewidth=linewidth)
ax_2depth.plot(level3_depths.index, level3_depths["('P2', 'depthN')"], label='level 3', color='red', alpha=0.6,linewidth=linewidth)
ax_2depth.plot(uncontrolled_depths.index, uncontrolled_depths["('P2', 'depthN')"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
# horizontal line at max depth, dotted red
ax_2depth.axhline(p2_max_depth, color='red', linestyle='--', label='Threshold',linewidth=linewidth)
#ax_2depth.set_ylabel('Depth [m]')
# remove the x ticks
ax_2depth.set_xticks([])
# title
ax_2depth.set_title('Depth in P2',y=0.8,fontsize='xx-large')

# plot flows out of p1
ax_1flow.plot(level1_flows.index, level1_flows["1"], label='level 1', color='blue', alpha=0.6,linewidth=linewidth)
ax_1flow.plot(level2_flows.index, level2_flows["1"], label='level 2', color='green', alpha=0.6,linewidth=linewidth)
ax_1flow.plot(level3_flows.index, level3_flows["1"], label='level 3', color='red', alpha=0.6,linewidth=linewidth)
ax_1flow.plot(uncontrolled_flows.index, uncontrolled_flows["1"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_1flow.set_ylabel('$m^3 / s$',rotation=0, fontsize='xx-large', labelpad = 25)

# title
ax_1flow.set_title('Flow out of P1',y=0.8,fontsize='xx-large')
ax_1flow.set_xticks([])

# plot flows out of p2
ax_2flows.plot(level1_flows.index, level1_flows["2"], label='level 1', color='blue', alpha=0.6,linewidth=linewidth)
ax_2flows.plot(level2_flows.index, level2_flows["2"], label='level 2', color='green', alpha=0.6,linewidth=linewidth)
ax_2flows.plot(level3_flows.index, level3_flows["2"], label='level 3', color='red', alpha=0.6,linewidth=linewidth)
ax_2flows.plot(uncontrolled_flows.index, uncontrolled_flows["2"], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
#ax_2flows.set_ylabel('Flow [m3/s]')
# title
ax_2flows.set_title('Flow out of P2',y=0.8,fontsize='xx-large')
ax_2flows.set_xticks([])

# plot flows out of p3
ax_8flows.plot(uncontrolled_data_log['simulation_time'], uncontrolled_data_log['flow']['8'], label='uncontrolled', color='black', alpha=0.6,linewidth=linewidth)
ax_8flows.plot(level1_data_log['simulation_time'], level1_data_log['flow']['8'], label='level 1', color='blue', alpha=0.6,linewidth=linewidth)
ax_8flows.plot(level2_data_log['simulation_time'], level2_data_log['flow']['8'], label='level 2', color='green', alpha=0.6,linewidth=linewidth)
ax_8flows.plot(level3_data_log['simulation_time'], level3_data_log['flow']['8'], label='level 3', color='red', alpha=0.6,linewidth=linewidth)
ax_8flows.set_ylabel('$m^3 / s$',rotation=0, fontsize='xx-large', labelpad = 25)

# title
ax_8flows.set_title('Flow at 8',y=0.8,fontsize='xx-large')

ax_8flows.axhline(flow_threshold, color='red', linestyle='--', label='Threshold',alpha=0.6,linewidth=linewidth)

unc_perf = sum(uncontrolled_data_log['performance_measure'])
level1_perf = sum(level1_data_log['performance_measure'])
level2_perf = sum(level2_data_log['performance_measure'])
level3_perf = sum(level3_data_log['performance_measure'])

perfstr = "Cost Difference from Uncontrolled\nLevel 1 = {:+.1%}\nLevel 2 = {:+.1%}\nLevel 3 = {:+.1%}".format((level1_perf - unc_perf)/unc_perf, (level2_perf - unc_perf)/unc_perf, (level3_perf - unc_perf)/unc_perf)
ax_8flows.annotate(perfstr, xy=(0.7, 0.5), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')


ax_8flows.tick_params(axis='both', labelsize='x-large')
ax_1depth.tick_params(axis='both', labelsize='x-large')
ax_2depth.tick_params(axis='both', labelsize='x-large')
ax_1flow.tick_params(axis='both', labelsize='x-large')
ax_2flows.tick_params(axis='both', labelsize='x-large')

fig.suptitle("Scenario Theta | Version " + version + " | Control: " + control, fontsize='xx-large')
plt.tight_layout()
plt.savefig("v" + version + "/" + str(control) + "_compare_levels.png", dpi=300)
plt.savefig("v" + version + "/" + str(control) + "_compare_levels.svg", dpi=300)
plt.show()


