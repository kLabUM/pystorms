'''
# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pystorms'])
subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
'''
import pystorms # this will be the first line of the program when dev is done

import numpy as np
import matplotlib.pyplot as plt
import pyswmm # requires pyswmm < 2.0.0
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
# THETA
mode = "compare" # "optimize" or "compare" (optimize is to use BOUC, compare is to look across different optimization methods)
evaluating = "constant-flow" # "constant-flow" or "efd" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
# level should always be 1 when optimizing parameters. controllers will be evaluated but not calibrated on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))

Cd = 1.0 # same for both valves
Ao = 1.0 # area is one square meter
g = 9.81 # m / s^2

# evaluating a given set of parameters for their cost
def run_swmm(constant_flows, efd_parameters=None,verbose=False):

    env = pystorms.scenarios.theta(version=version)
    

    env.env.sim.start()
    done = False
    
    max_depths_array = np.array([])
    max_depths = dict()
    peak_filling_degrees = np.array([])
    for state in env.config['states']:
        if 'depth' in state[1]:
            node_id = state[0]
            max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)
            peak_filling_degrees = np.append(peak_filling_degrees, 0.0)
            #print(node_id, max_depths[node_id]) # to check
    #kprint(max_depths)
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    u_open_pct = np.ones((len(env.config['action_space']),1))*1 # begin open
    
    while not done:
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state()
            # update peak filing degrees
            for idx in range(len(state)):
                if state[idx]/max_depths_array[idx] > peak_filling_degrees[idx]:
                    peak_filling_degrees[idx] = state[idx]/max_depths_array[idx]

            for idx in range(len(u_open_pct)): # set opening percentage to achieve the desired flow rate
       
                # flow rate for an orifice is Q = CA sqrt(2gh)
                # assume this scales linearly with opening percentage
                Q_desired = constant_flows[idx]
                if state[idx] < 1e-3:
                    u_open_pct[idx,0] = 1.0
                else:
                    u_open_pct[idx,0] = Q_desired / (Cd*Ao*np.sqrt(2*g*state[idx]))
                # bound the opening percentage to [0,1]
                if u_open_pct[idx,0] > 1:
                    u_open_pct[idx,0] = 1
                elif u_open_pct[idx,0] < 0:
                    u_open_pct[idx,0] = 0
                
            if evaluating == "constant-flow":
                done = env.step(u_open_pct.flatten())
            elif evaluating == "efd":
                filling_degrees = np.array([pyswmm.Nodes(env.env.sim)[node_id].depth/max_depths[node_id] for node_id in max_depths.keys()]).reshape(-1,1)
                u_diff = filling_degrees - np.mean(filling_degrees) 
                u_open_pct = u_open_pct + efd_parameters*u_diff
                #u_avg = np.mean(u_open_pct)
                #u_diff = u_avg - u_open_pct

                #u_open_pct = u_open_pct + efd_parameters[0]*u_diff 
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0] > 1:
                        u_open_pct[i,0] = 1
                    elif u_open_pct[i,0] < 0:
                        u_open_pct[i,0] = 0
                done = env.step(u_open_pct.flatten())

            else:
                print("error. control scenario not recongized.")
                done = True
                
            if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0: 
                u_print = u_open_pct.flatten()
                y_measured = env.state().reshape(-1,1)
                print("              y_measured,  u")
                print(np.c_[np.array(env.config['states']),np.round(y_measured,2) , np.round(u_print.reshape(-1,1),3)])
                print("current time, end time")
                print(env.env.sim.current_time, env.env.sim.end_time)
                print("\n")
            
            if (not done) and (env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1)):
                final_depths = env.state()
                
        else:
            done = env.step(u_open_pct.flatten())
            

    return {"data_log": env.data_log, "final_depths": final_depths,"peak_filling_degrees":peak_filling_degrees}

# --- Shared evaluator for objective/constraint with caching ---
def evaluate_cf_point(params_array):
    """
    Compute objective and constraint once, cache into Sim_cf.computed_return_values,
    and return a dict with 'objective' and 'constraint'.
    """
    key = tuple(np.array(params_array).flatten())
    if key in Sim_cf.computed_return_values:
        return Sim_cf.computed_return_values[key]

    # Run the simulation once
    data = run_swmm(np.array(params_array).flatten(), None, verbose=False)

    # Objective (same formula as Sim_cf.objective)
    flow_cost = 0.0
    for _, value in data['data_log']['flow'].items():
        flow_exceed = [x - 0.25 for x in value]
        flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
        flow_cost += sum(flow_exceed)
    objective_cost = float(flow_cost + sum(data['final_depths']) + np.std(data['final_depths']))

    # Constraint (same formula as Sim_cf.constraint)
    flood_cost = 0.0
    for _, value in data['data_log']['flooding'].items():
        flood_cost += sum(value)
    if 0.0 < flood_cost < 1.0:
        flood_cost = 1.0
    elif flood_cost <= 0.0:
        flood_cost = float(max(data['peak_filling_degrees']))
    else:
        flood_cost = float(flood_cost)

    Sim_cf.computed_return_values[key] = {
        'objective': objective_cost,
        'constraint': flood_cost
    }
    return Sim_cf.computed_return_values[key]

class Sim_cf:
    threshold = 0.99 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()

    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            res = evaluate_cf_point(sample.numpy())
            return_values.append(res['objective'])
        return np.array(return_values).reshape(-1, 1)

    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            res = evaluate_cf_point(sample.numpy())
            return_values.append(res['constraint'])
        return np.array(return_values).reshape(-1, 1)
    '''
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_cf.computed_return_values.keys():
                return_values.append(Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                constant_flow_params = np.array(sample).flatten()    
                data = run_swmm(constant_flow_params, None,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_exceed = [x - 0.25 for x in value]
                    flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
                    flow_cost += sum(flow_exceed)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0 and flood_cost < 1:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'])

                #pystorms_cost = sum(data['data_log']['performance_measure'])

                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                #Sim_cf.computed_return_values[tuple(sample.numpy())]['pystorms_cost'] = pystorms_cost
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
                constant_flow_params = np.array(sample).flatten()    
                data = run_swmm(constant_flow_params, None,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_exceed = [x - 0.25 for x in value]
                    flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
                    flow_cost += sum(flow_exceed)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0 and flood_cost < 1:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'])

                #pystorms_cost = sum(data['data_log']['performance_measure'])
                    
                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                #Sim_cf.computed_return_values[tuple(sample.numpy())]['pystorms_cost'] = pystorms_cost
                return_values.append(flood_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
'''
    
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
    
def observer_cf(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_cf.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_cf.constraint(query_points)),
        }

class Sim_efd:
    threshold = 0.99 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()
    
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_efd.computed_return_values.keys():
                return_values.append(Sim_efd.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                efd_params = np.array(sample).flatten()[-1]
                constant_flows = np.array(sample).flatten()[:-1]
                data = run_swmm(constant_flows, efd_params,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_exceed = [x - 0.25 for x in value]
                    flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
                    flow_cost += sum(flow_exceed)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0 and flood_cost < 1:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'])

                Sim_efd.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_efd.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_efd.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                return_values.append(objective_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_efd.computed_return_values.keys():
                return_values.append(Sim_efd.computed_return_values[tuple(sample.numpy())]['constraint'])
            else:
                efd_params = np.array(sample).flatten()[-1]
                constant_flows = np.array(sample).flatten()[:-1]   
                data = run_swmm(constant_flows, efd_params,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_exceed = [x - 0.25 for x in value]
                    flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
                    flow_cost += sum(flow_exceed)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0 and flood_cost < 1:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'])
                    
                Sim_efd.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_efd.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_efd.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                return_values.append(flood_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    

def observer_efd(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_efd.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_efd.constraint(query_points)),
        }


def create_bo_model(data):
        gpr = build_gpr(data, search_space, likelihood_variance = 1e-7)
        return GaussianProcessRegression(gpr)


if evaluating == "constant-flow" and mode == "optimize":
    lower_bounds = []
    upper_bounds = []
    for i in range(1, 3):  
        lower_bounds.append(0.01)
        upper_bounds.append(1.0)
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 100
    initial_data = observer_cf(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 200
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_cf, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_cf.threshold)[0]
    # if feasible indices is empty, then there are no feasible points
    if len(feasible_indices) == 0:
        print("No feasible points found.")
    else:
        # find the index of the best feasible query point (tf doesn't play nice with np argmin)
        best_feasible_index = -1
        for idx in feasible_indices:
            if best_feasible_index == -1:
                best_feasible_index = idx
            elif data[OBJECTIVE].observations[idx] < data[OBJECTIVE].observations[best_feasible_index]:
                best_feasible_index = idx
        #best_feasible_index = feasible_indices[np.argmin(data[OBJECTIVE].observations[feasible_indices])]
        # get the best feasible query point
        best_feasible_point = data[OBJECTIVE].query_points[best_feasible_index]
        # get the best feasible observation
        best_feasible_observation = data[OBJECTIVE].observations[best_feasible_index]


    # save the optimal constant heads and the entire optimization object
    np.savetxt(str("v" +version +"/optimal_constant_flows.txt"), best_feasible_point.numpy())
    np.savetxt(str("v" +version +"/optimal_constant_flows_cost.txt"), best_feasible_observation.numpy())
    # save the whole object
    with open("bo_constant_flows.pkl", "wb") as f:
        pickle.dump(opt_result, f)


    # plot the model and observations of the objective function
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    # query points is 2 dimensional, so make a heatmap with observations as the z variable
    # locations will be: data['OBJECITVE'].query_points
    # z values will be: data['OBJECTIVE'].observations
    # plot the observations
    ax[0].scatter(data[OBJECTIVE].query_points[:, 0], data[OBJECTIVE].query_points[:,1], c=data[OBJECTIVE].observations, label="observations")


    # plot the GP model outputs across a grid of points
    # make a grid of two-dimensional points across the search space
    x1 = np.linspace(0.01, 1.0, 100)
    x2 = np.linspace(0.01, 1.0, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1, X2], axis=-1)
    objective_predicted = models['OBJECTIVE'].predict_y(X)
    objective_mean_predicted = objective_predicted[0]
    # plot the heatmap with the predicted values, and the observations on top
    # objective_mean_predicted is (100,100,1) so need to reshape to (100,100)
    tf.experimental.numpy.experimental_enable_numpy_behavior() # to allow reshape
    c = ax[0].contourf(X1, X2, objective_mean_predicted.reshape(100,100), alpha=0.2, label="model")

    ax[0].tick_params(axis='both', labelsize='large')
    ax[1].tick_params(axis='both', labelsize='large')
    
    # colorbar
    plt.colorbar(c, ax=ax[0])
    ax[0].set_title("Objective",fontsize='xx-large')
    ax[0].set_xlabel("Constant Flow 1",fontsize='x-large')
    ax[0].set_ylabel("Constant\nFlow 2",rotation=0,labelpad=35,fontsize='x-large')

    # plot the constraint observations and model on the right
    ax[1].scatter(data[CONSTRAINT].query_points[:, 0], data[CONSTRAINT].query_points[:,1], c=data[CONSTRAINT].observations, label="observations")
    constraint_predicted = models[CONSTRAINT].predict_y(X)
    constraint_mean_predicted = constraint_predicted[0]
    c = ax[1].contourf(X1, X2, constraint_mean_predicted.reshape(100,100), alpha=0.2, label="model")
    # add a black contour line to show the boundary of the feasible region
    ax[1].contour(X1, X2, constraint_mean_predicted.reshape(100,100), levels=[Sim_cf.threshold], colors='black')
    # put it on the objective function plot too
    ax[0].contour(X1, X2, constraint_mean_predicted.reshape(100,100), levels=[Sim_cf.threshold], colors='black')
    cbar = plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Constraint",fontsize='xx-large')
    ax[1].set_xlabel("Constant Flow 1",fontsize='x-large')
    plt.tight_layout()
    plt.savefig(str("v" +version + "/constrained_bo_cf.png"))
    plt.savefig(str("v" +version + "/constrained_bo_cf.svg"))
    #plt.show()
    plt.close('all')
    
elif evaluating == "efd" and mode == "optimize":

    lower_bounds = []
    upper_bounds = []
    for i in range(1, 3):
        lower_bounds.append(0.01)
        upper_bounds.append(1.0)
        
    lower_bounds.append(0.0)
    upper_bounds.append(1.0)
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 25
    initial_data = observer_efd(search_space.sample(num_initial_points))

    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 250
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_efd, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_cf.threshold)[0]
    # if feasible indices is empty, then there are no feasible points
    if len(feasible_indices) == 0:
        print("No feasible points found.")
    else:
        # find the index of the best feasible query point (tf doesn't play nice with np argmin)
        best_feasible_index = -1
        for idx in feasible_indices:
            if best_feasible_index == -1:
                best_feasible_index = idx
            elif data[OBJECTIVE].observations[idx] < data[OBJECTIVE].observations[best_feasible_index]:
                best_feasible_index = idx
        #best_feasible_index = feasible_indices[np.argmin(data[OBJECTIVE].observations[feasible_indices])]
        # get the best feasible query point
        best_feasible_point = data[OBJECTIVE].query_points[best_feasible_index]
        # get the best feasible observation
        best_feasible_observation = data[OBJECTIVE].observations[best_feasible_index]


    # save the optimal constant heads and the entire optimization object
    np.savetxt(str("v" +version +"/optimal_efd_params.txt"), best_feasible_point.numpy())
    np.savetxt(str("v" +version +"/optimal_efd_params_cost.txt"), best_feasible_observation.numpy())
    # save the whole object
    #with open("bo_efd_params.pkl", "wb") as f:
    #    pickle.dump(opt_result, f)




elif evaluating == 'both' and mode == "optimize":
    evaluating="constant-flow"
    domain = []
    
    
    lower_bounds = []
    upper_bounds = []
    for i in range(1, 3):

        
        lower_bounds.append(0.01)
        upper_bounds.append(1.0)
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 100
    initial_data = observer_cf(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 300
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_cf, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_cf.threshold)[0]
    # if feasible indices is empty, then there are no feasible points
    if len(feasible_indices) == 0:
        print("No feasible points found.")
    else:
        # find the index of the best feasible query point (tf doesn't play nice with np argmin)
        best_feasible_index = -1
        for idx in feasible_indices:
            if best_feasible_index == -1:
                best_feasible_index = idx
            elif data[OBJECTIVE].observations[idx] < data[OBJECTIVE].observations[best_feasible_index]:
                best_feasible_index = idx
        #best_feasible_index = feasible_indices[np.argmin(data[OBJECTIVE].observations[feasible_indices])]
        # get the best feasible query point
        best_feasible_point = data[OBJECTIVE].query_points[best_feasible_index]
        # get the best feasible observation
        best_feasible_observation = data[OBJECTIVE].observations[best_feasible_index]


    # save the optimal constant heads and the entire optimization object
    np.savetxt(str("v" +version +"/optimal_constant_flows.txt"), best_feasible_point.numpy())
    np.savetxt(str("v" +version +"/optimal_constant_flows_cost.txt"), best_feasible_observation.numpy())
    # save the whole object
    with open("bo_constant_flows.pkl", "wb") as f:
        pickle.dump(opt_result, f)


    # plot the model and observations of the objective function
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    # query points is 2 dimensional, so make a heatmap with observations as the z variable
    # locations will be: data['OBJECITVE'].query_points
    # z values will be: data['OBJECTIVE'].observations
    # plot the observations
    ax[0].scatter(data[OBJECTIVE].query_points[:, 0], data[OBJECTIVE].query_points[:,1], c=data[OBJECTIVE].observations, label="observations")


    # plot the GP model outputs across a grid of points
    # make a grid of two-dimensional points across the search space
    x1 = np.linspace(0.01, 1.0, 100)
    x2 = np.linspace(0.01, 1.0, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.stack([X1, X2], axis=-1)
    objective_predicted = models['OBJECTIVE'].predict_y(X)
    objective_mean_predicted = objective_predicted[0]
    # plot the heatmap with the predicted values, and the observations on top
    # objective_mean_predicted is (100,100,1) so need to reshape to (100,100)
    tf.experimental.numpy.experimental_enable_numpy_behavior() # to allow reshape
    c = ax[0].contourf(X1, X2, objective_mean_predicted.reshape(100,100), alpha=0.2, label="model")

    
    # colorbar
    plt.colorbar(c, ax=ax[0])
    ax[0].set_title("Objective observations and gp model")
    ax[0].set_xlabel("Constant Flow 1")
    ax[0].set_ylabel("Constant Flow 2")

    # plot the constraint observations and model on the right
    ax[1].scatter(data[CONSTRAINT].query_points[:, 0], data[CONSTRAINT].query_points[:,1], c=data[CONSTRAINT].observations, label="observations")
    constraint_predicted = models[CONSTRAINT].predict_y(X)
    constraint_mean_predicted = constraint_predicted[0]
    c = ax[1].contourf(X1, X2, constraint_mean_predicted.reshape(100,100), alpha=0.2, label="model")
    # add a black contour line to show the boundary of the feasible region
    ax[1].contour(X1, X2, constraint_mean_predicted.reshape(100,100), levels=[Sim_cf.threshold], colors='black')
    # put it on the objective function plot too
    ax[0].contour(X1, X2, constraint_mean_predicted.reshape(100,100), levels=[Sim_cf.threshold], colors='black')
    plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Constraint observations and gp model")
    ax[1].set_xlabel("Constant Flow 1")
    plt.savefig(str("v" +version + "/constrained_bo_cf.png"))
    plt.savefig(str("v" +version + "/constrained_bo_cf.svg"))
    plt.close('all')
    



    evaluating = "efd"
    
    lower_bounds = []
    upper_bounds = []
    for i in range(1, 3):
        lower_bounds.append(0.01)
        upper_bounds.append(1.0)
        
    lower_bounds.append(0.0)
    upper_bounds.append(1.0)
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 225
    initial_data = observer_efd(search_space.sample(num_initial_points))

    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 350
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_efd, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_cf.threshold)[0]
    # if feasible indices is empty, then there are no feasible points
    if len(feasible_indices) == 0:
        print("No feasible points found.")
    else:
        # find the index of the best feasible query point (tf doesn't play nice with np argmin)
        best_feasible_index = -1
        for idx in feasible_indices:
            if best_feasible_index == -1:
                best_feasible_index = idx
            elif data[OBJECTIVE].observations[idx] < data[OBJECTIVE].observations[best_feasible_index]:
                best_feasible_index = idx
        #best_feasible_index = feasible_indices[np.argmin(data[OBJECTIVE].observations[feasible_indices])]
        # get the best feasible query point
        best_feasible_point = data[OBJECTIVE].query_points[best_feasible_index]
        # get the best feasible observation
        best_feasible_observation = data[OBJECTIVE].observations[best_feasible_index]


    # save the optimal constant heads and the entire optimization object
    np.savetxt(str("v" +version +"/optimal_efd_params.txt"), best_feasible_point.numpy())
    np.savetxt(str("v" +version +"/optimal_efd_params_cost.txt"), best_feasible_observation.numpy())
    # save the whole object
    with open("bo_efd_params.pkl", "wb") as f:
        pickle.dump(opt_result, f)


elif evaluating == "constant-flow" and mode == "compare":
    print("\nComparing optimization methods for constant flow\n")
    
    # Common setup for all methods
    lower_bounds = [0.01, 0.01]
    upper_bounds = [1.0, 1.0]
    search_space = Box(lower_bounds, upper_bounds)
    
    # Import required libraries for additional optimization methods
    from scipy.optimize import dual_annealing, differential_evolution
    import time
    from trieste.acquisition.function import ExpectedImprovement
    import matplotlib.ticker as mticker
    
    # For tracking function calls
    function_call_count = 0
    
    # Define a combined objective function for methods that don't separate constraint and objective
    def combined_objective(params):
        global function_call_count
        function_call_count += 1

        params_array = np.array(params).flatten()
        res = evaluate_cf_point(params_array)
        # If the constraint is violated, return a large penalty
        if res['constraint'] > Sim_cf.threshold:
            return 1e6
        return float(res['objective'])

    # For tracking performance across methods
    results = {
        "BOUC": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
        "BO": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
        "DA": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
        "DE": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')}
    }
    
    # Generate common initial points for all methods
    num_initial_points = 25  # Number of initial points (5 is the min for differential evolution)
    num_steps = 50#200  # Reduced for comparison
    initial_seed = 42  # Use fixed seed for reproducibility
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
    bouc_combined_costs = []  # Store combined costs for fair comparison
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
    models = opt_result.try_get_final_models()
    bouc_fcalls = len(obj_data.query_points)
    # Process BOUC results and get combined costs for each point
    bouc_combined_costs = []
    for i in range(len(obj_data.query_points)):
        point = obj_data.query_points[i].numpy()

        # Use the same combined objective used by the other methods
        combined_cost = combined_objective(point)
        bouc_combined_costs.append(combined_cost)

        # Record original BOUC objective/constraint
        obj_value = float(obj_data.observations[i][0])
        con_value = float(con_data.observations[i][0])

        # Track best feasible point according to BOUC's objective
        if con_value <= Sim_cf.threshold and obj_value < best_feasible_obj:
            best_feasible_obj = obj_value
            best_feasible_point = point
            best_feasible_idx = i

        bouc_costs.append(obj_value)
        bouc_fcalls_arr.append(i + 1)

    # Store results for comparison
    results["BOUC"]["costs"] = bouc_costs
    results["BOUC"]["combined_costs"] = bouc_combined_costs
    results["BOUC"]["fcalls"] = bouc_fcalls_arr

    # Use the combined cost for the best point
    if best_feasible_point is not None:
        results["BOUC"]["best_params"] = best_feasible_point
        results["BOUC"]["best_cost"] = bouc_combined_costs[best_feasible_idx]
    else:
        idx = int(np.argmin(bouc_combined_costs))
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
        gpr = build_gpr(data, search_space, likelihood_variance = 1e-7)
        return GaussianProcessRegression(gpr)
    initial_data_bo = observer_vanilla_bo(initial_points)
    '''
    for i in range(len(initial_data_bo.query_points)):
        value = float(initial_data_bo.observations[i][0])
        bo_costs.append(value)
        bo_fcalls_arr.append(i + 1)
    '''
    # ^ those will be counted as part of the final_data. double counting to track them here.
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
    csv_path = f"v{version}/optimal_constant_flows_summary.csv"
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
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls.png")
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls.svg")

    # --- Create a zoomed-in plot ---
    plt.figure(figsize=(12, 8))
    
    # Find the best cost achieved across all methods
    min_cost_all = float('inf')
    for method in results.keys():
        if results[method]["best_cost_so_far"] and min(results[method]["best_cost_so_far"]) < min_cost_all:
            min_cost_all = min(results[method]["best_cost_so_far"])
    
    # Set y-limits from slightly below best cost to twice the best cost
    #plt.ylim(0.95 * min_cost_all, 2.0 * min_cost_all)
    
    for method, data in results.items():
        if data["fcalls"] and data["best_cost_so_far"]:
            plt.plot(data["fcalls"], data["best_cost_so_far"], 'o-', label=labels[method], markersize=8)
    
    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('Best Cost Found', fontsize=14)
    plt.title('Optimization Methods Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls_zoom.png")
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls_zoom.svg")
    plt.show()

        # --- BOUC GP estimate: objective contour + feasibility boundary + best points ---
    try:
        # Grid across the search space
        x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
        y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
        X1, X2 = np.meshgrid(x, y)
        X = np.stack([X1, X2], axis=-1)

        # Predict objective and constraint with BOUC GP models
        objective_predicted = models[OBJECTIVE].predict_y(X)
        objective_mean_predicted = objective_predicted[0]
        constraint_predicted = models[CONSTRAINT].predict_y(X)
        constraint_mean_predicted = constraint_predicted[0]

        # Allow reshape on TF tensors
        tf.experimental.numpy.experimental_enable_numpy_behavior()

        plt.figure(figsize=(10, 8))
        cp = plt.contourf(
            X1, X2,
            objective_mean_predicted.reshape(100, 100),
            levels=50, cmap='viridis'
        )
        plt.colorbar(cp)
        # Feasibility boundary from BOUC constraint GP
        plt.contour(
            X1, X2,
            constraint_mean_predicted.reshape(100, 100),
            levels=Sim_cf.threshold, colors='black', linestyles='solid'
        )

        # Plot each method's best point
        for method, data in results.items():
            if data["best_params"] is not None:
                plt.plot(
                    data["best_params"][0], data["best_params"][1],
                    'x', label=f'{method} Best', markersize=12
                )

        plt.title('BOUC GP Objective Estimate with Feasibility Boundary')
        plt.xlabel('Constant Flow 1')
        plt.ylabel('Constant Flow 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"v{version}/bouc_gp_objective_with_best_points.png")
        plt.savefig(f"v{version}/bouc_gp_objective_with_best_points.svg")
        plt.show()
    except Exception as e:
        print(f"Skipping BOUC GP contour plot due to: {e}")
