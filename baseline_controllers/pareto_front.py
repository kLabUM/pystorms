from msilib import Control
import pystorms # this will be the first line of the program when dev is done
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

network = "epsilon"
version = "2" # of the model
level = "1" # noise and faults

# create separate plots for each control scenario
for control_scenario in ["constant-flow","equal-filling"]:
    # create a multilevel dictionary with parameter values as the highest keys
    parameter_values = []
    data = {}
    # find the csv files which match our network, version, and level
    folder_path = str("./" + network + "/v" + version + "/lev" + level + "/results")
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and control_scenario in file and "=" in file and "costs" in file:
            # the parameter value will be a float (including a decimal) between "=" and ".csv"
            parameter = float(file.split("=")[1].split(".csv")[0])
            parameter_values.append( parameter)       
            data[parameter] = pd.read_csv(str(folder_path + "/" + file))
            data[parameter]["average_ending_filling_degree"] = np.mean(data[parameter]["final_filling"])
            data[parameter]["cost"] = data[parameter]["cost"].iloc[0]
        
    # print data by its keys and values
    #for key in data.keys():
    #    print(key)
    #    print(data[key])
    
    uncontrolled_data = pd.read_csv(str("./epsilon/v" + version + "/results/costs_uncontrolled.csv"))
    uncontrolled_data["average_ending_filling_degree"] = np.mean(uncontrolled_data["final_filling"])

    color_norm = colors.Normalize(vmin=min(parameter_values),vmax=max(parameter_values))

    # plot the pareto front by cost on the x axis, average ending filling degree on the y axis, and parameter value as the color of the circle
    plt.figure(figsize=(6,6))
    for parameter_value in parameter_values:
        plt.scatter(data[parameter_value]["cost"].iloc[0], data[parameter_value]["average_ending_filling_degree"].iloc[0], 
                    c=parameter_value, cmap='viridis', norm = color_norm)

    plt.xlabel("Cost")
    plt.ylabel("Average Ending\nFilling Degree", rotation=0, labelpad=40)
    plt.title(str(network + " | " + control_scenario + "\nversion " + version + " | level " + level))
    max_cost = max([data[parameter_value]["cost"].iloc[0] for parameter_value in parameter_values])
    max_cost = max(max_cost , uncontrolled_data["cost"].iloc[0])
    max_filling = max([data[parameter_value]["average_ending_filling_degree"].iloc[0] for parameter_value in parameter_values])
    max_filling = max(max_filling, uncontrolled_data["average_ending_filling_degree"].iloc[0])
    plt.xlim(-100, max_cost+100)
    plt.ylim(0.0,max_filling+0.05 )
    # grid on
    plt.grid()
    plt.colorbar()

    # black square for the uncontrolled result
    plt.scatter(uncontrolled_data["cost"].iloc[0], uncontrolled_data["average_ending_filling_degree"].iloc[0], c='black', marker='s', s=100)

    plt.tight_layout()
    
    # save the figure
    plt.savefig(str("./" + network + "/v" + version + "/lev" + level + "/pareto_front_" + control_scenario + ".png"))
    plt.savefig(str("./" + network + "/v" + version + "/lev" + level + "/pareto_front_" + control_scenario + ".svg"))
    plt.show()
    
plt.close('all')


# plot both baseline controllers together
for control_scenario in ["constant-flow","equal-filling"]:
    parameter_values = []
    data = {}
    # find the csv files which match our network, version, and level
    folder_path = str("./" + network + "/v" + version + "/lev" + level + "/results")
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and control_scenario in file and "=" in file and "costs" in file:
            # the parameter value will be a float (including a decimal) between "=" and ".csv"
            parameter = float(file.split("=")[1].split(".csv")[0])
            parameter_values.append( parameter)       
            data[parameter] = pd.read_csv(str(folder_path + "/" + file))
            data[parameter]["average_ending_filling_degree"] = np.mean(data[parameter]["final_filling"])
            data[parameter]["cost"] = data[parameter]["cost"].iloc[0]
    
    if control_scenario == 'constant-flow':
        constant_flow_data = data
    elif control_scenario == "equal-filling":
        equal_filling_data = data
        
# plot the pareto front by cost on the x axis, average ending filling degree on the y axis, and parameter value as the color of the circle
plt.figure(figsize=(6,6))
for key in constant_flow_data.keys():
    plt.scatter(constant_flow_data[key]["cost"].iloc[0], constant_flow_data[key]["average_ending_filling_degree"].iloc[0], c='r')
for key in equal_filling_data.keys():
    plt.scatter(equal_filling_data[key]["cost"].iloc[0], equal_filling_data[key]["average_ending_filling_degree"].iloc[0], c='b')

plt.xlabel("Cost")
# make the x axis log scale
#plt.xscale('log')
#plt.ylabel("Average Ending\nFilling Degree")
# rotate the ylabel to be vertical
plt.ylabel("Average Ending\nFilling Degree", rotation=0, labelpad=40)
plt.title(str(network + "\nversion " + version + " | level " + level))
#plt.colorbar(label=label)
# rotate the colorbar label to be vertical
# set the xlim to be between 0 and whatever the maximum cost is
max_cost = max([constant_flow_data[key]["cost"].iloc[0] for key in constant_flow_data.keys()])
max_cost = max(max_cost , uncontrolled_data["cost"].iloc[0])
max_filling = max([equal_filling_data[key]["average_ending_filling_degree"].iloc[0] for key in equal_filling_data.keys()])
max_filling = max(max_filling, uncontrolled_data["average_ending_filling_degree"].iloc[0])
   
# grid on
plt.grid()
# create a patch for the legend. modpods is blue, constant-head is red
import matplotlib.patches as mpatches
constant_flow_patch = mpatches.Patch(color='red', label='Constant Flow')
efd_patch = mpatches.Patch(color='blue', label='Equal Filling Degree')


# vertical line at the uncontrolled cost (7.69*10^3)
#plt.axvline(x=7.69*10**3, color='k', linestyle='--', label='uncontrolled cost')
# plot a big black square for the uncontrolled result
plt.scatter(uncontrolled_data["cost"].iloc[0], uncontrolled_data["average_ending_filling_degree"].iloc[0], c='black', marker='s', s=100)

# add the uncontrolled cost to the legend
uncontrolled_patch = mpatches.Patch(color='black', linestyle='--', label='Uncontrolled')
plt.legend(handles=[constant_flow_patch, efd_patch,uncontrolled_patch])

plt.tight_layout()

plt.savefig(str("./" + network + "/v" + version + "/lev" + level + "/pareto_front_both.png"))
plt.savefig(str("./" + network + "/v" + version + "/lev" + level + "/pareto_front_both.svg"))


plt.show()




