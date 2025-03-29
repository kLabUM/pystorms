import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import os
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

scenarios = ["theta","alpha","gamma","delta","epsilon"]

# create a figure with 5 subplots, one for each scenario. stacked horizontally.
fig, axs = plt.subplots(1, len(scenarios), figsize=(12, 6), sharey=False)

# the leftmost bar chart is for theta, the next for alpha, and so on.
theta_performance = pd.DataFrame(columns = ["Uncontrolled","Constant Flow","Equal Filling"],index=["Level 1","Level 2","Level 3"])
uncontrolled_data_log = pd.read_pickle("theta/v2/uncontrolled_level_1_data_log.pkl")
theta_performance.loc[:,"Uncontrolled"] = sum(uncontrolled_data_log["performance_measure"])
cf_lev1 = pd.read_pickle("theta/v2/constant-flow_level_1_data_log.pkl")
cf_lev2 = pd.read_pickle("theta/v2/constant-flow_level_2_data_log.pkl")
cf_lev3 = pd.read_pickle("theta/v2/constant-flow_level_3_data_log.pkl")
theta_performance.loc["Level 1","Constant Flow"] = sum(cf_lev1["performance_measure"])
theta_performance.loc["Level 2","Constant Flow"] = sum(cf_lev2["performance_measure"])
theta_performance.loc["Level 3","Constant Flow"] = sum(cf_lev3["performance_measure"])
ef_lev1 = pd.read_pickle("theta/v2/equal-filling_level_1_data_log.pkl")
ef_lev2 = pd.read_pickle("theta/v2/equal-filling_level_2_data_log.pkl")
ef_lev3 = pd.read_pickle("theta/v2/equal-filling_level_3_data_log.pkl")
theta_performance.loc["Level 1","Equal Filling"] = sum(ef_lev1["performance_measure"])
theta_performance.loc["Level 2","Equal Filling"] = sum(ef_lev2["performance_measure"])
theta_performance.loc["Level 3","Equal Filling"] = sum(ef_lev3["performance_measure"])
theta_performance = theta_performance.astype(float)

'''
theta_performance.plot(kind='bar',ax=axs[0],color = ['black','green','blue'])

axs[0].set_ylabel("Theta\n\nCost",rotation='horizontal',labelpad = 35)
# make the xtick labels horizontal
axs[0].set_xticklabels(theta_performance.index, rotation=0)
# put the legend off the plot to the right
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
'''
# make a line plot of the columns of theta_performance with the index as the xticks
theta_performance.plot(kind='line',ax=axs[0],color = ['black','green','blue'], marker = 'o', linewidth = 2)
axs[0].set_title("Theta")
axs[0].set_ylabel("Cost",rotation='horizontal',labelpad = 25)
# make the legend fontsize small
axs[0].legend(loc='best', fontsize='x-small')

# format to print 3 sig figs, scientific notation
pd.options.display.float_format = '{:,.3g}'.format
# for all the columns except "Uncontrolled", replace the value with a tuple of the current value and the percentage change from the uncontrolled value
for col in theta_performance.columns[1:]:
    theta_performance[col] = theta_performance.apply(lambda x: (x[col], (x[col] - x["Uncontrolled"]) / x["Uncontrolled"] * 100), axis=1)
# print the theta_performance dataframe with the percentage change (including the trailing % sign)
theta_performance = theta_performance.map(lambda x: f"{x[0]:,.3g} ({x[1]:.1f}%)" if isinstance(x, tuple) else x)
# also format the uncontrolled performance
theta_performance["Uncontrolled"] = theta_performance["Uncontrolled"].map(lambda x: f"{x:,.3g}")
# name the index "theta"
theta_performance.index.name = "Theta"
print(theta_performance)
# save this as a csv file
theta_performance.to_csv("theta_performance_summary.csv")

# now do the same for alpha
alpha_performance = pd.DataFrame(columns = ["Uncontrolled","Structural","Constant Flow","Equal Filling"],index=["Level 1","Level 2","Level 3"])
uncontrolled_data_log = pd.read_pickle("alpha/v2/results/uncontrolled_data_log.pkl")
alpha_performance.loc[:,"Uncontrolled"] = sum(uncontrolled_data_log["performance_measure"])
structural_data_log = pd.read_pickle("alpha/v2/results/structural_data_log.pkl")
alpha_performance.loc[:,"Structural"] = sum(structural_data_log["performance_measure"])
cf_lev1 = pd.read_pickle("alpha/v2/lev1/results/constant-flow_param=0.0_data_log.pkl")
cf_lev2 = pd.read_pickle("alpha/v2/lev2/results/constant-flow_param=0.0_data_log.pkl")
cf_lev3 = pd.read_pickle("alpha/v2/lev3/results/constant-flow_param=0.0_data_log.pkl")
alpha_performance.loc["Level 1","Constant Flow"] = sum(cf_lev1["performance_measure"])
alpha_performance.loc["Level 2","Constant Flow"] = sum(cf_lev2["performance_measure"])
alpha_performance.loc["Level 3","Constant Flow"] = sum(cf_lev3["performance_measure"])
ef_lev1 = pd.read_pickle("alpha/v2/lev1/results/equal-filling_param=0.0_data_log.pkl")
ef_lev2 = pd.read_pickle("alpha/v2/lev2/results/equal-filling_param=0.0_data_log.pkl")
ef_lev3 = pd.read_pickle("alpha/v2/lev3/results/equal-filling_param=0.0_data_log.pkl")
alpha_performance.loc["Level 1","Equal Filling"] = sum(ef_lev1["performance_measure"])
alpha_performance.loc["Level 2","Equal Filling"] = sum(ef_lev2["performance_measure"])
alpha_performance.loc["Level 3","Equal Filling"] = sum(ef_lev3["performance_measure"])
alpha_performance = alpha_performance.astype(float)

# create a bar chart for alpha
'''
alpha_performance.plot(kind='bar',ax=axs[1],color = ['black','orange','green','blue'])
axs[1].set_ylabel("Alpha\n\nCost",rotation='horizontal',labelpad = 35)
# make the xtick labels horizontal
axs[1].set_xticklabels(alpha_performance.index, rotation=0)
# put the legend off the plot to the right
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
'''
# make a line plot of the columns of alpha_performance with the index as the xticks
alpha_performance.plot(kind='line',ax=axs[1],color = ['black','orange','green','blue'], marker = 'o', linewidth = 2)
axs[1].set_title("Alpha")
axs[1].set_ylabel("Cost",rotation='horizontal',labelpad = 25)
# make the legend fontsize small
axs[1].legend(loc='best', fontsize='x-small')


# for all the columns except "Uncontrolled", replace the value with a tuple of the current value and the percentage change from the uncontrolled value
for col in alpha_performance.columns[1:]:
    alpha_performance[col] = alpha_performance.apply(lambda x: (x[col], (x[col] - x["Uncontrolled"]) / x["Uncontrolled"] * 100), axis=1)
# print the alpha_performance dataframe with the percentage change (including the trailing % sign)
alpha_performance = alpha_performance.map(lambda x: f"{x[0]:,.3g} ({x[1]:.1f}%)" if isinstance(x, tuple) else x)
# also format the uncontrolled performance 
alpha_performance["Uncontrolled"] = alpha_performance["Uncontrolled"].map(lambda x: f"{x:,.3g}")
# name the index "alpha"
alpha_performance.index.name = "Alpha"
print(alpha_performance)
# save this as a csv file
alpha_performance.to_csv("alpha_performance_summary.csv")

# make the table for gamma, don't bother with the plot
gamma_performance = pd.DataFrame(columns = ["Uncontrolled","Constant Flow","Equal Filling"],index=["Level 1","Level 2","Level 3"])
uncontrolled_data_log = pd.read_pickle("gamma/v2/results/uncontrolled_data_log.pkl")
gamma_performance.loc[:,"Uncontrolled"] = sum(uncontrolled_data_log["performance_measure"])
cf_lev1 = pd.read_pickle("gamma/v2/lev1/results/constant-flow_param=0.0_data_log.pkl")
cf_lev2 = pd.read_pickle("gamma/v2/lev2/results/constant-flow_param=0.0_data_log.pkl")
cf_lev3 = pd.read_pickle("gamma/v2/lev3/results/constant-flow_param=0.0_data_log.pkl")
gamma_performance.loc["Level 1","Constant Flow"] = sum(cf_lev1["performance_measure"])
gamma_performance.loc["Level 2","Constant Flow"] = sum(cf_lev2["performance_measure"])
gamma_performance.loc["Level 3","Constant Flow"] = sum(cf_lev3["performance_measure"])
ef_lev1 = pd.read_pickle("gamma/v2/lev1/results/equal-filling_param=0.0_data_log.pkl")
ef_lev2 = pd.read_pickle("gamma/v2/lev2/results/equal-filling_param=0.0_data_log.pkl")
ef_lev3 = pd.read_pickle("gamma/v2/lev3/results/equal-filling_param=0.0_data_log.pkl")
gamma_performance.loc["Level 1","Equal Filling"] = sum(ef_lev1["performance_measure"])
gamma_performance.loc["Level 2","Equal Filling"] = sum(ef_lev2["performance_measure"])
gamma_performance.loc["Level 3","Equal Filling"] = sum(ef_lev3["performance_measure"])
gamma_performance = gamma_performance.astype(float)

# for all the columns except "Uncontrolled", replace the value with a tuple of the current value and the percentage change from the uncontrolled value
for col in gamma_performance.columns[1:]:
    gamma_performance[col] = gamma_performance.apply(lambda x: (x[col], (x[col] - x["Uncontrolled"]) / x["Uncontrolled"] * 100), axis=1)
# print the gamma_performance dataframe with the percentage change (including the trailing % sign)
gamma_performance = gamma_performance.map(lambda x: f"{x[0]:,.3g} ({x[1]:.1f}%)" if isinstance(x, tuple) else x)
# also format the uncontrolled performance
gamma_performance["Uncontrolled"] = gamma_performance["Uncontrolled"].map(lambda x: f"{x:,.3g}")
# name the index "gamma"
gamma_performance.index.name = "Gamma"
print(gamma_performance)
# save this as a csv file
gamma_performance.to_csv("gamma_performance_summary.csv")

# now do the same for delta
delta_performance = pd.DataFrame(columns = ["Uncontrolled","Static Plus Rule","Prop Outflow"],index=["Level 1","Level 2","Level 3"])
uncontrolled_data_log = pd.read_pickle("delta/v2/results/uncontrolled_data_log.pkl")
delta_performance.loc[:,"Uncontrolled"] = sum(uncontrolled_data_log["performance_measure"])
spr1_data_log = pd.read_pickle("delta/v2/lev1/results/static-plus-rule_param=0.0_data_log.pkl")
spr2_data_log = pd.read_pickle("delta/v2/lev2/results/static-plus-rule_param=0.0_data_log.pkl")
spr3_data_log = pd.read_pickle("delta/v2/lev3/results/static-plus-rule_param=0.0_data_log.pkl")
delta_performance.loc["Level 1","Static Plus Rule"] = sum(spr1_data_log["performance_measure"])
delta_performance.loc["Level 2","Static Plus Rule"] = sum(spr2_data_log["performance_measure"])
delta_performance.loc["Level 3","Static Plus Rule"] = sum(spr3_data_log["performance_measure"])
po_lev1 = pd.read_pickle("delta/v2/lev1/results/prop-outflow_param=0.0_data_log.pkl")
po_lev2 = pd.read_pickle("delta/v2/lev2/results/prop-outflow_param=0.0_data_log.pkl")
po_lev3 = pd.read_pickle("delta/v2/lev3/results/prop-outflow_param=0.0_data_log.pkl")
delta_performance.loc["Level 1","Prop Outflow"] = sum(po_lev1["performance_measure"])
delta_performance.loc["Level 2","Prop Outflow"] = sum(po_lev2["performance_measure"])
delta_performance.loc["Level 3","Prop Outflow"] = sum(po_lev3["performance_measure"])
delta_performance = delta_performance.astype(float)

# for all the columns except "Uncontrolled", replace the value with a tuple of the current value and the percentage change from the uncontrolled value
for col in delta_performance.columns[1:]:
    delta_performance[col] = delta_performance.apply(lambda x: (x[col], (x[col] - x["Uncontrolled"]) / x["Uncontrolled"] * 100), axis=1)
# print the delta_performance dataframe with the percentage change (including the trailing % sign)
delta_performance = delta_performance.map(lambda x: f"{x[0]:,.3g} ({x[1]:.1f}%)" if isinstance(x, tuple) else x)
# also format the uncontrolled performance
delta_performance["Uncontrolled"] = delta_performance["Uncontrolled"].map(lambda x: f"{x:,.3g}")
# name the index "delta"
delta_performance.index.name = "Delta"
print(delta_performance)
# save this as a csv file
delta_performance.to_csv("delta_performance_summary.csv")

epsilon_performance = pd.DataFrame(columns = ["Uncontrolled","Constant Flow","Equal Filling"],index=["Level 1","Level 2","Level 3"])
uncontrolled_data_log = pd.read_pickle("epsilon/v2/results/uncontrolled_data_log.pkl")
epsilon_performance.loc[:,"Uncontrolled"] = sum(uncontrolled_data_log["performance_measure"])
cf_lev1 = pd.read_pickle("epsilon/v2/lev1/results/constant-flow_param=0.0_data_log.pkl")
cf_lev2 = pd.read_pickle("epsilon/v2/lev2/results/constant-flow_param=0.0_data_log.pkl")
cf_lev3 = pd.read_pickle("epsilon/v2/lev3/results/constant-flow_param=0.0_data_log.pkl")
epsilon_performance.loc["Level 1","Constant Flow"] = sum(cf_lev1["performance_measure"])
epsilon_performance.loc["Level 2","Constant Flow"] = sum(cf_lev2["performance_measure"])
epsilon_performance.loc["Level 3","Constant Flow"] = sum(cf_lev3["performance_measure"])
ef_lev1 = pd.read_pickle("epsilon/v2/lev1/results/equal-filling_param=0.0_data_log.pkl")
ef_lev2 = pd.read_pickle("epsilon/v2/lev2/results/equal-filling_param=0.0_data_log.pkl")
ef_lev3 = pd.read_pickle("epsilon/v2/lev3/results/equal-filling_param=0.0_data_log.pkl")
epsilon_performance.loc["Level 1","Equal Filling"] = sum(ef_lev1["performance_measure"])
epsilon_performance.loc["Level 2","Equal Filling"] = sum(ef_lev2["performance_measure"])
epsilon_performance.loc["Level 3","Equal Filling"] = sum(ef_lev3["performance_measure"])
epsilon_performance = epsilon_performance.astype(float)

for col in epsilon_performance.columns[1:]:
    epsilon_performance[col] = epsilon_performance.apply(lambda x: (x[col], (x[col] - x["Uncontrolled"]) / x["Uncontrolled"] * 100), axis=1)
# print the epsilon_performance dataframe with the percentage change (including the trailing % sign)
epsilon_performance = epsilon_performance.map(lambda x: f"{x[0]:,.3g} ({x[1]:.1f}%)" if isinstance(x, tuple) else x)
# also format the uncontrolled performance
epsilon_performance["Uncontrolled"] = epsilon_performance["Uncontrolled"].map(lambda x: f"{x:,.3g}")
# name the index "epsilon"
epsilon_performance.index.name = "Epsilon"
print(epsilon_performance)
# save this as a csv file
epsilon_performance.to_csv("epsilon_performance_summary.csv")
    


# concatenate the performance summary csv files (not the dataframes which are currently in memory)
import fileinput
# Directory containing the CSV files
directory = os.getcwd()

# Output file
output_file = 'performance_summary.csv'

# Get list of CSV files in the directory
csv_files = ['theta_performance_summary.csv','alpha_performance_summary.csv',
             'gamma_performance_summary.csv','delta_performance_summary.csv', 'epsilon_performance_summary.csv']

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    for i, line in enumerate(fileinput.input(files=csv_files)):
        outfile.write(line)
        # record the max width (number of commas) so far
        if i == 0:
            max_width = line.count(",")
        else:
        # if the current line has more commas, update max_width
            if line.count(",") > max_width:
                max_width = line.count(",")    
        




