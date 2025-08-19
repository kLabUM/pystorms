# this script uses very cheap to evaluate constraint and objective functions to ensure optimization comparison is done correctly.

import numpy as np
import matplotlib.pyplot as plt
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
import dill as pickle
# print current working directory
import os
print(os.getcwd())
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# imports for joint optimization of expected improvement and probability of feasibility (https://secondmind-labs.github.io/trieste/4.3.0/notebooks/inequality_constraints.html)
import tensorflow as tf
from trieste.space import Box
import trieste
from trieste.data import Dataset
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization

class Sim_cf:
    threshold = 0.0 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()
    
    @staticmethod
    def objective(input_data): # the objective is the 2D ackley function
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_cf.computed_return_values.keys():
                return_values.append(Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                objective_cost = ackley(sample.numpy())
                constraint_value = disk(sample.numpy())

                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_value
                return_values.append(objective_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_cf.computed_return_values.keys():
                return_values.append(Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'])
            else:
                objective_cost = ackley(sample.numpy())
                constraint_value = disk(sample.numpy())

                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_value
                return_values.append(constraint_value)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
    
def observer_cf(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_cf.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_cf.constraint(query_points)),
        }

def create_bo_model(data):
        gpr = build_gpr(data, search_space)
        return GaussianProcessRegression(gpr)

def disk(sample): # constraint function
    return ((sample[0]-2)**2 + sample[1]**2 - 2)*(25 - ((sample[0]-2)**2 + sample[1]**2)) + 0.1*sample[0]**3 - 0.2*sample[1]**3

def ackley(sample): # objective function
    return -20*np.exp(-0.2*np.sqrt(0.5*(sample[0]**2 + sample[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*sample[0]) + np.cos(2*np.pi*sample[1]))) + 20 + np.exp(1)

# For tracking function calls
function_call_count = 0

def combined_objective(sample):
    global function_call_count
    function_call_count += 1
    # This function is used to evaluate the combined objective and constraint
    objective_cost = ackley(sample)
    constraint_value = disk(sample)
    # if the constraint is violated, return a large penalty
    if constraint_value > Sim_cf.threshold:
        return 1e6
    return objective_cost



# Common setup for all methods
lower_bounds = [-10.0, -10.0]
upper_bounds = [10.0, 10.0]
search_space = Box(lower_bounds, upper_bounds)
    
# Import required libraries for additional optimization methods
from scipy.optimize import dual_annealing, differential_evolution
import time
from trieste.acquisition.function import ExpectedImprovement
import matplotlib.ticker as mticker
    

# For tracking performance across methods
results = {
    "BOUC": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
    "BO": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
    "DA": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
    "DE": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')}
}
    
# Generate common initial points for all methods
num_initial_points = 25  # Number of initial points (5 is the min for differential evolution)
num_steps = 50  # Reduced for comparison
initial_seed = 7  # Use fixed seed for reproducibility
tf.random.set_seed(initial_seed)
np.random.seed(initial_seed)
    
# Generate initial points that all methods will use
initial_points = search_space.sample(num_initial_points)
initial_points_np = initial_points.numpy()
    
# 1. Bayesian Optimization with Unknown Constraints (BOUC) using Trieste
print("Running BOUC optimization...")
function_call_count = 0  # Reset counter
initial_data = observer_cf(initial_points)
bouc_fcalls = 0 # initial points are counted in the length of query_points
obj_data = initial_data[OBJECTIVE]
con_data = initial_data[CONSTRAINT]
best_feasible_obj = float('inf')
best_feasible_point = None
bouc_costs = []  # Original BOUC objective values
bouc_combined_costs = []  # Store pystorms costs for fair comparison
bouc_fcalls_arr = []

# Run optimization
initial_models = trieste.utils.map_values(create_bo_model, initial_data)
pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
eci = trieste.acquisition.ExpectedConstrainedImprovement(
    OBJECTIVE, pof.using(CONSTRAINT)
)
rule = EfficientGlobalOptimization(eci)
    
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_cf, search_space)
bouc_start_time = time.time()
opt_result = bo.optimize(
    num_steps, initial_data, initial_models, rule
)
bouc_total_time = time.time() - bouc_start_time
datasets = opt_result.try_get_final_datasets()
obj_data = datasets[OBJECTIVE]
con_data = datasets[CONSTRAINT]
bouc_fcalls = len(obj_data.query_points)
    # Process BOUC results and get pystorms costs for each point
for i in range(len(obj_data.query_points)):
    point = obj_data.query_points[i].numpy()

    combined_cost = combined_objective(point)
    bouc_combined_costs.append(combined_cost)
        
    # Record original BOUC objective value
    obj_value = float(obj_data.observations[i][0])
    con_value = float(con_data.observations[i][0])
        
    # Track best feasible point according to BOUC's objective
    if con_value <= Sim_cf.threshold and obj_value < best_feasible_obj:
        best_feasible_obj = obj_value
        best_feasible_point = point
        best_feasible_idx = i  # Store the index here for later reference
            
    bouc_costs.append(obj_value)
    bouc_fcalls_arr.append(i + 1)

# Store results for comparison
results["BOUC"]["costs"] = bouc_costs
results["BOUC"]["combined_costs"] = bouc_combined_costs
results["BOUC"]["fcalls"] = bouc_fcalls_arr
    
# Use the already calculated pystorms cost for the best point
if best_feasible_point is not None:
    results["BOUC"]["best_params"] = best_feasible_point
    results["BOUC"]["best_cost"] = bouc_combined_costs[best_feasible_idx]
else:
    # If no feasible point, use the best objective point
    idx = np.argmin(obj_data.observations)
    results["BOUC"]["best_params"] = obj_data.query_points[idx].numpy()
    results["BOUC"]["best_cost"] = bouc_combined_costs[idx]

print("BOUC function calls:", bouc_fcalls_arr)
print("BOUC combined costs:", bouc_combined_costs)
    
# 2. Vanilla Bayesian Optimization using Trieste
print("\nRunning vanilla BO...")
function_call_count = 0  # Reset counter
bo_fcalls = 0
bo_costs = []
bo_fcalls_arr = []
class Sim_vanilla_bo:
    computed_return_values = dict()
    @staticmethod
    def objective(input_data):
        global bo_fcalls
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_vanilla_bo.computed_return_values.keys():
                return_values.append(Sim_vanilla_bo.computed_return_values[tuple(sample.numpy())])
            else:
                bo_fcalls += 1
                value = combined_objective(sample.numpy())
                Sim_vanilla_bo.computed_return_values[tuple(sample.numpy())] = value
                return_values.append(value)
        return np.array(return_values).reshape(-1, 1)
def observer_vanilla_bo(query_points):
    return Dataset(query_points, Sim_vanilla_bo.objective(query_points))
def vanilla_bo_create_model(data):
    gpr = build_gpr(data, search_space)
    return GaussianProcessRegression(gpr)
initial_data_bo = observer_vanilla_bo(initial_points)
'''
for i in range(len(initial_data_bo.query_points)):
    value = float(initial_data_bo.observations[i][0])
    bo_costs.append(value)
    bo_fcalls_arr.append(i + 1)
'''
# ^ those will be counted as part of the final_data. it would be double counting to track them here.
initial_model_bo = vanilla_bo_create_model(initial_data_bo)
ei = ExpectedImprovement()
rule_bo = EfficientGlobalOptimization(ei)
bo_vanilla = trieste.bayesian_optimizer.BayesianOptimizer(observer_vanilla_bo, search_space)
bo_start_time = time.time()
opt_result_bo = bo_vanilla.optimize(
    num_steps, initial_data_bo, initial_model_bo, rule_bo
)
bo_total_time = time.time() - bo_start_time
final_data_bo = opt_result_bo.try_get_final_dataset()
for i in range(len(final_data_bo.query_points)):
    value = float(final_data_bo.observations[i][0])
    bo_costs.append(value)
    bo_fcalls_arr.append(i + 1)
best_idx = np.argmin(final_data_bo.observations)
best_point = final_data_bo.query_points[best_idx].numpy()
best_value = float(final_data_bo.observations[best_idx].numpy()[0])
results["BO"]["costs"] = bo_costs
results["BO"]["fcalls"] = bo_fcalls_arr
results["BO"]["best_params"] = best_point
results["BO"]["best_cost"] = best_value
print("BO function calls:", bo_fcalls_arr)
print("BO costs:", bo_costs)
    
# 3. Dual Annealing
print("\nRunning Dual Annealing optimization...")
function_call_count = 0  # Reset counter
bounds_da = list(zip(lower_bounds, upper_bounds))
da_costs = []
da_fcalls_arr = []
x0_costs = []
for i in range(len(initial_points_np)):
    x0_costs.append(combined_objective(initial_points_np[i]))
    da_costs.append(x0_costs[-1])
    da_fcalls_arr.append(i + 1)
best_idx = np.argmin(x0_costs)
x0 = initial_points_np[best_idx]
da_best_cost = x0_costs[best_idx]
global da_best_params
da_best_params = x0.copy()
def da_callback(x, f, context):
    global da_best_params
    da_fcalls_arr.append(function_call_count)
    if f < da_costs[-1]:
        da_best_params = x.copy()
        da_costs.append(f)
    else:
        da_costs.append(da_costs[-1])
            
    return False
    
# Run optimization
da_max_calls = num_steps
da_max_iter = num_steps // 3 # max calls is a soft limit. max iter is hard.

da_start_time = time.time()
res_da = dual_annealing(combined_objective, bounds=bounds_da, 
                        maxfun=da_max_calls, maxiter=da_max_iter, callback=da_callback,
                        x0=x0, no_local_search=True)
da_total_time = time.time() - da_start_time
if da_costs[-1] != res_da.fun or da_fcalls_arr[-1] != function_call_count:
    da_fcalls_arr.append(function_call_count)
    da_costs.append(res_da.fun)
    print(f"Added final DA state: cost={res_da.fun}, fcalls={function_call_count}")
results["DA"]["costs"] = da_costs
results["DA"]["fcalls"] = da_fcalls_arr
results["DA"]["best_params"] = res_da.x
results["DA"]["best_cost"] = res_da.fun
print("DA function calls:", da_fcalls_arr)
print("DA costs:", da_costs)
    
# 4. Differential Evolution
print("\nRunning Differential Evolution optimization...")
function_call_count = 0  # Reset counter
bounds_de = list(zip(lower_bounds, upper_bounds))
de_costs = []
de_fcalls_arr = []
x0_costs = []
for i in range(len(initial_points_np)):
    x0_costs.append(combined_objective(initial_points_np[i]))
    de_costs.append(x0_costs[-1])
    de_fcalls_arr.append(i + 1)
best_idx = np.argmin(x0_costs)
x0 = initial_points_np[best_idx]
de_best_cost = x0_costs[best_idx]
global de_best_params
de_best_params = x0.copy()
def de_callback(x, convergence):
    global de_best_params, function_call_count, de_costs, de_fcalls_arr
    de_fcalls_arr.append(function_call_count)
    f = combined_objective(x)
    function_call_count -= 1  # Decrement to avoid double counting
    if f < de_costs[-1]:
        de_best_params = x.copy()
        de_costs.append(f)
    else:
        de_costs.append(de_costs[-1])
            
    return False
    
# Create initial population that includes our initial points
popsize = num_initial_points
population = np.array(initial_points_np)
    
# Set maxiter to ensure comparable number of function evaluations
de_max_iter = 2*max(1, num_steps // (popsize * len(lower_bounds)))
#  The maximum number of function evaluations (with no polishing) is: (maxiter + 1) * popsize * (N - N_equal)
print(f"DE max iterations: {de_max_iter}, popsize: {popsize}, function evaluations: {(de_max_iter + 1) * popsize * (len(lower_bounds) - 1)}")
    
    
de_start_time = time.time()
# Run optimization
res_de = differential_evolution(
    combined_objective, 
    bounds=bounds_de, 
    maxiter=de_max_iter, 
    callback=de_callback,
    popsize=popsize,
    init=population,
    polish=False
)
de_total_time = time.time() - de_start_time
if de_costs[-1] != res_de.fun or de_fcalls_arr[-1] != function_call_count:
    de_fcalls_arr.append(function_call_count)
    de_costs.append(res_de.fun)
    print(f"Added final DE state: cost={res_de.fun}, fcalls={function_call_count}")
results["DE"]["costs"] = de_costs
results["DE"]["fcalls"] = de_fcalls_arr
results["DE"]["best_params"] = res_de.x
results["DE"]["best_cost"] = res_de.fun
print("DE function calls:", de_fcalls_arr)
print("DE costs:", de_costs)
    
# Compare results
print("\nOptimization Results Comparison (using consistent performance measure):")
print("Starting costs (from pystorms performance measure):")
print(f"  BOUC: {bouc_combined_costs[0]:.4f}, BO: {bo_costs[0]:.4f}, DA: {da_costs[0]:.4f}, DE: {de_costs[0]:.4f}")
print("-" * 80)
print(f"{'Method':<10} | {'Best Cost':<15} | {'Function Calls':<15} | {'Time (s)':<15}")
print("-" * 80)
for method, data in results.items():
    if method == "BOUC":
        total_time = bouc_total_time
        fcalls = max(data["fcalls"]) if data["fcalls"] else 0
    elif method == "BO":
        total_time = bo_total_time
        fcalls = max(data["fcalls"]) if data["fcalls"] else 0
    elif method == "DA":
        total_time = da_total_time
        fcalls = max(data["fcalls"]) if data["fcalls"] else 0
    elif method == "DE":
        total_time = de_total_time
        fcalls = max(data["fcalls"]) if data["fcalls"] else 0
            
    if data["best_params"] is not None:
        print(f"{method:<10} | {data['best_cost']:<15.4f} | {fcalls:<15} | {total_time:<15.2f}")
print("-" * 80)
'''
# Save best parameters to files
for method, data in results.items():
    if data["best_params"] is not None:
        np.savetxt(f"v{version}/optimal_constant_flows_{method}.txt", data["best_params"])
        np.savetxt(f"v{version}/optimal_constant_flows_cost_{method}.txt", [data["best_cost"]])
'''
import csv

# Save best parameters and costs for all methods into a single CSV
csv_path = f"benchmark_dev/optimal_summary.csv"
param_count = len(lower_bounds)
header = [
    "method",
    "best_cost",
    "optimization_time_sec",
    "num_function_calls"
] + [f"param_{i+1}" for i in range(param_count)]

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for method, data in results.items():
        if data["best_params"] is not None:
            # Get timing and function call info
            if method == "BOUC":
                total_time = bouc_total_time
                fcalls = max(data["fcalls"]) if data["fcalls"] else 0
            elif method == "BO":
                total_time = bo_total_time
                fcalls = max(data["fcalls"]) if data["fcalls"] else 0
            elif method == "DA":
                total_time = da_total_time
                fcalls = max(data["fcalls"]) if data["fcalls"] else 0
            elif method == "DE":
                total_time = de_total_time
                fcalls = max(data["fcalls"]) if data["fcalls"] else 0
            else:
                total_time = 0
                fcalls = 0
            params = np.array(data["best_params"]).flatten()
            row = [method, data["best_cost"], total_time, fcalls]
            row.extend(params)
            writer.writerow(row)
print(f"Saved summary CSV to {csv_path}")
labels = {
    "BOUC": f"BOUC ({bouc_total_time/60.0:.1f}min)",
    "BO": f"BO ({bo_total_time/60.0:.1f}min)",
    "DA": f"DA ({da_total_time/60.0:.1f}min)",
    "DE": f"DE ({de_total_time/60.0:.1f}min)"
}        
    
# --- Rolling minimum (best cost so far) arrays for convergence plots ---

def rolling_min(costs, feasible=None):
    best = []
    min_so_far = float('inf')
    for i, c in enumerate(costs):
        if feasible is not None:
            if feasible[i]:
                min_so_far = min(min_so_far, c)
        else:
            min_so_far = min(min_so_far, c)
        best.append(min_so_far)
    return best

# For BOUC, only update best if feasible
bouc_feasible = []
for i in range(len(results["BOUC"]["costs"])):
    # You have con_data available for BOUC
    if i < len(con_data.observations):
        feasible = float(con_data.observations[i][0]) <= Sim_cf.threshold
    else:
        feasible = False
    bouc_feasible.append(feasible)
# For BOUC, use pystorms cost directly 
results["BOUC"]["best_cost_so_far"] = rolling_min(results["BOUC"]["combined_costs"])

# For other methods, all are assumed feasible
for method in ["BO", "DA", "DE"]:
    results[method]["best_cost_so_far"] = rolling_min(results[method]["costs"])

# --- Plotting ---
plt.figure(figsize=(12, 8))
# y axis log scale
plt.yscale('log')


for method, data in results.items():
    if data["fcalls"] and data["best_cost_so_far"]:
        plt.plot(data["fcalls"], data["best_cost_so_far"], 'o-', label=labels[method], markersize=8)
plt.xlabel('Function Evaluations', fontsize=14)
plt.ylabel('Best Cost Found', fontsize=14)
plt.title('Optimization Methods Comparison by Function Evaluations', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig(f"benchmark_dev/optimization_methods_comparison_by_fcalls.png")
plt.savefig(f"benchmark_dev/optimization_methods_comparison_by_fcalls.svg")
plt.show()

# --- Create a zoomed-in plot ---
plt.figure(figsize=(12, 8))
    
# Find the best cost achieved across all methods
min_cost_all = float('inf')
for method in results.keys():
    if results[method]["best_cost_so_far"] and min(results[method]["best_cost_so_far"]) < min_cost_all:
        min_cost_all = min(results[method]["best_cost_so_far"])
    
# Set y-limits from slightly below best cost to twice the best cost
plt.ylim(0.95 * min_cost_all, 2.0 * min_cost_all)
    
for method, data in results.items():
    if data["fcalls"] and data["best_cost_so_far"]:
        plt.plot(data["fcalls"], data["best_cost_so_far"], 'o-', label=labels[method], markersize=8)
    
plt.xlabel('Function Evaluations', fontsize=14)
plt.ylabel('Best Cost Found', fontsize=14)
plt.title('Optimization Methods Comparison (Zoomed)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize='x-large')
plt.tight_layout()
plt.savefig(f"benchmark_dev/optimization_methods_comparison_by_fcalls_zoom.png")
plt.savefig(f"benchmark_dev/optimization_methods_comparison_by_fcalls_zoom.svg")
plt.show()

# plot the true objective function and each optimization's best point

x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
X, Y = np.meshgrid(x, y)
Z = ackley([X, Y])
# include the disk constraint boundary
Z_disk = disk([X, Y])


plt.figure(figsize=(10, 8))
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
# just draw lines where the constraint is zero
plt.contour(X, Y, Z_disk, levels=[0], colors='black', linestyles='solid')
#plt.scatter(initial_points_np[:, 0], initial_points_np[:, 1], c='grey', label='Initial Points', s=20, alpha=0.5)
    
# Plot each method's best point
for method, data in results.items():
    if data["best_params"] is not None:
        plt.plot(data["best_params"][0], data["best_params"][1], 'o', label=f'{method} Best', markersize=10)
    
plt.title('Objective Function Contour Plot with Best Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.tight_layout()
plt.savefig(f"benchmark_dev/objective_function_with_best_points.png")
plt.savefig(f"benchmark_dev/objective_function_with_best_points.svg")
plt.show()


