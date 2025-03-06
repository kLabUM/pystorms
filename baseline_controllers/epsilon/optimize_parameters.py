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
import pyswmm
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

# imports for trieste active learning of feasibility region (https://secondmind-labs.github.io/trieste/4.3.0/notebooks/feasible_sets.html)

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
import tensorflow as tf
from trieste.space import Box
import trieste
from trieste.data import Dataset
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization

# EPSILON
evaluating = "efd" # "constant-flow" or "efd" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
level = "1"
# level should always be 1 when optimizing parameters. controllers will be evaluated but not optimized on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))


# evaluating a given set of parameters for their cost
def run_swmm(constant_flows, efd_parameters=None,verbose=False):
    env = pystorms.scenarios.epsilon(version=version,level="1")

    env.env.sim.start()
    done = False

    # per https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf the equation for flow over a transverse weir 
    # (all control assets in epsilon are transverse weirs) is:
    # Q = Cw L h^(1.5)
    # where Q is the flow, Cw is the weir discharge coefficient, L is the length of the weir, and h is the head over the weir
    # modpods will pick up the constants in each case. so we just need to provide the weir head to the correct power
    # the maximum hiehgt of the weir varies by asset (all in feet, project is in imperial units)
    H = {"ISD001": 14.7, "ISD002": 9.0, "ISD003": 14.0, "ISD004": 15.5, "ISD005": 15.5, "ISD006": 15.5, "ISD007": 15.5, "ISD008": 12.25, "ISD009": 15.5, "ISD010": 10.5, "ISD011": 11.5}
    H_array = [14.7,9.0,14.0, 15.5 , 15.5 ,15.5, 15.5 ,12.25,15.5,10.5 ,11.5]
    max_depths_array = np.array([])
    max_depths = dict()
    peak_filling_degrees = dict()
    for state in env.config['states']:
        if 'depth' in state[1]:
            node_id = state[0]
            max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)
            peak_filling_degrees[node_id] = 0.0
            #print(node_id, max_depths[node_id]) # to check
    #print(max_depths)
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    u_open_pct = np.ones((len(env.config['action_space']),1))*1 # begin open

    while not done:
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state()[:11] # only the first eleven, controlled basins
            for idx in range(len(u_open_pct)): # set h_weir to achieve the desired head over the weir
       
                max_weir_height = H[env.config['action_space'][idx]]

                #h_up = min(y_measured[idx,0],max_weir_height) # upstream storage basin depth
                h_up = state[idx] # the depth in the junction just above the weir
                if constant_flows[idx] >= h_up: # all the way open
                    u_open_pct[idx] = 1.0
                else: # somewhere in between
                    h_weir = h_up - constant_flows[idx] # desired weir position
                    closed_percentage = h_weir / max_weir_height
                    u_open_pct[idx] = 1 - closed_percentage
                u_open_pct[idx] = max(0.0,u_open_pct[idx]) # don't allow negative opening percentages
                u_open_pct[idx] = min(1.0,u_open_pct[idx]) # don't allow opening percentages greater than 1

            for idx, state in enumerate(env.config['states']):
                if 'depth' in state[1]:
                    node_id = state[0]
                    peak_filling_degrees[node_id] = max(peak_filling_degrees[node_id],env.state()[idx]/max_depths[node_id])
            if evaluating == "constant-flow":
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0]< 0.09:
                        u_open_pct[i,0] = 0.09
                done = env.step(u_open_pct.flatten())
            elif evaluating == "efd":
                # this is a slightly different formulation for equal filling degree
                # the fixed depth of the inline storages is based on the height of the weir
                # so by making the weirs more similar opening percentages, the fixed depths are converging
                u_avg = np.mean(u_open_pct)
                u_diff = u_avg - u_open_pct
                for i in [0,3,8]: # the most downstream dams
                    TSS_conc = env.state()[-1]
                    delta_TSS = TSS_conc - 150 # 150 is roughly the long term average
                    u_diff[i,0] = efd_parameters[0]*delta_TSS # increase flow out of the outflow controllers when TSS is low. reduce when high.
                    # so efd_parameters[0] should be negative

                u_open_pct = u_open_pct + efd_parameters[1]*u_diff 
                #for i in range(len(u_open_pct)):
                #    if u_open_pct[i,0]< 0.09:
                #        print("efd using storage above the weir")
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0]< 0.0:
                        u_open_pct[i,0] = 0.0
                    elif u_open_pct[i,0]> 1.0:
                        u_open_pct[i,0] = 1.0
                done = env.step(u_open_pct.flatten())

            else:
                print("error. control scenario not recongized.")
                done = True
                
            if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0: 
                u_print = u_open_pct.flatten()
                y_measured = env.state().reshape(-1,1)
                print("              y_measured,  u")
                print(np.c_[np.array(env.config['states'][:11]),np.round(y_measured[:11],2) , np.round(u_print.reshape(-1,1),3)])
                print("current time, end time")
                print(env.env.sim.current_time, env.env.sim.end_time)
                print("\n")
            
            if env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
                final_depths = env.state()[:11]
                final_weir_settings = u_open_pct.flatten()
                
        else:
            done = env.step(u_open_pct.flatten())
            

    return {"data_log": env.data_log, "final_depths": final_depths,"peak_filling_degrees": peak_filling_degrees, "final_weir_settings":final_weir_settings}

'''
def f_constant_flows(constant_flows):
    # flatten the actions
    constant_head_params = np.array(constant_flows).flatten()    

    data = run_swmm(constant_head_params, None,verbose=False)
    return_value = data["cost"] + 100*sum(data["final_depths"]) + 100*np.std(data['final_depths'])
    return float(return_value)

def f_efd(efd_parameters):
    efd_params = np.array(efd_parameters).flatten()

    data = run_swmm(optimal_constant_flows, efd_params,verbose=False)
    return float(data['cost'] + 100*sum(data['final_depths'])) + 100*np.std(data['final_depths'])
'''

class Sim_cf:
    threshold = 0.99 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()
    
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_cf.computed_return_values.keys():
                return_values.append(Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                constant_flow_params = np.array(sample).flatten()    
                data = run_swmm(constant_flow_params, None,verbose=False)
                loading_cost = 0.0
                for key,value in data['data_log']['loading'].items():
                    loading_cost += sum(value)
                flow_cost = np.std(np.array(data['data_log']['flow']['001']).flatten()) # penalize the outflow variation
                
                #objective_cost = loading_cost + sum(data['final_depths'])*5e1 + np.std(data['final_depths'])*1e3
                objective_cost = np.std(data['final_weir_settings'])*5e4 + flow_cost*1e3
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'].values())

                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
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
                loading_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    loading_cost += sum(value)
                flow_cost = np.std(np.array(data['data_log']['flow']['001']).flatten()) # penalize the outflow variation
                
                #objective_cost = loading_cost + sum(data['final_depths'])*5e1 + np.std(data['final_depths'])*1e3
                objective_cost = np.std(data['final_weir_settings'])*5e4 + flow_cost*1e3
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'].values())

                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                return_values.append(flood_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values

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
                constant_flow_params = np.array(sample).flatten()[:-2]
                efd_params = np.array(sample).flatten()[-2:]
                data = run_swmm(constant_flow_params, efd_params,verbose=False)
                loading_cost = 0.0
                for key,value in data['data_log']['loading'].items():
                    loading_cost += sum(value)
                flow_cost = np.std(np.array(data['data_log']['flow']['001']).flatten()) # penalize the outflow variation
                
                #objective_cost = loading_cost + sum(data['final_depths'])*5e1 + np.std(data['final_depths'])*1e3
                #objective_cost = loading_cost + np.std(data['final_weir_settings'])*2e5 + flow_cost*1e2
                #objective_cost = loading_cost  + flow_cost*1e2
                objective_cost = loading_cost + np.std(data['final_weir_settings'])*5e4 + flow_cost*1e3
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
                    flood_cost = 1.0
                elif any(np.array(data['final_weir_settings']) < 0.01): # weirs ending completely closed
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'].values())

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
                constant_flow_params = np.array(sample).flatten()[:-2]
                efd_params = np.array(sample).flatten()[-2:]
                data = run_swmm(constant_flow_params, efd_params,verbose=False)
                loading_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    loading_cost += sum(value)
                flow_cost = np.std(np.array(data['data_log']['flow']['001']).flatten()) # penalize the outflow variation
                
                #objective_cost = loading_cost + sum(data['final_depths'])*5e1 + np.std(data['final_depths'])*1e3
                #objective_cost = loading_cost + np.std(data['final_weir_settings'])*2e5 + flow_cost*1e2
                #objective_cost = loading_cost  + flow_cost*1e2
                objective_cost = loading_cost + np.std(data['final_weir_settings'])*5e4 + flow_cost*1e3
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
                    flood_cost = 1.0
                elif any(np.array(data['final_weir_settings']) < 0.01): # weirs ending completely closed
                    flood_cost = 1.0
                elif flood_cost <= 0.0:
                    flood_cost = max(data['peak_filling_degrees'].values())

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
    gpr = build_gpr(data, search_space)
    return GaussianProcessRegression(gpr)
    
if evaluating == "constant-flow":
    lower_bounds = []
    upper_bounds = []
    for i in range(11):
        lower_bounds.append(0.7)
        upper_bounds.append(4.0)

        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 2
    initial_data = observer_cf(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 1
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



elif evaluating == "efd":
    lower_bounds = []
    upper_bounds = []
    for i in range(11):
        lower_bounds.append(0.7)
        upper_bounds.append(4.0)
    lower_bounds.append(-1e-1) # tss feedback
    upper_bounds.append(-1e-4)
    lower_bounds.append(0.0) # efd gain
    upper_bounds.append(3.0)
        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 120
    initial_data = observer_efd(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_efd.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 450
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_efd, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_efd.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_efd.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_efd_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        with open("bo_efd.pkl", "wb") as f:
            pickle.dump(opt_result, f)
    
elif evaluating == 'both':
    evaluating = "constant-flow"
    lower_bounds = []
    upper_bounds = []
    for i in range(11):
        lower_bounds.append(0.7)
        upper_bounds.append(4.0)

        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 100
    initial_data = observer_cf(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 350
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
        #with open("bo_constant_flows.pkl", "wb") as f:
        #    pickle.dump(opt_result, f)


    evaluating = "efd"
    lower_bounds = []
    upper_bounds = []
    for i in range(11):
        lower_bounds.append(0.7)
        upper_bounds.append(4.0)
    lower_bounds.append(-1e-1) # tss feedback
    upper_bounds.append(-1e-4)
    lower_bounds.append(0.0) # efd gain
    upper_bounds.append(3.0)
        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 120
    initial_data = observer_efd(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_efd.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 450
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_efd, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_efd.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_efd.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_efd_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        #with open("bo_efd.pkl", "wb") as f:
        #    pickle.dump(opt_result, f)