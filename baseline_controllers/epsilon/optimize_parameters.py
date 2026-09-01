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
mode = "compare" # "optimize" or "compare"
evaluating = "constant-flow" # "constant-flow" or "efd" or "both"
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
                    if u_open_pct[i,0]< 0.09:
                        u_open_pct[i,0] = 0.09
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
    
if evaluating == "constant-flow" and mode == "optimize":
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



elif evaluating == "efd" and mode == "optimize":
    lower_bounds = list(np.loadtxt(str("v" +version +"/optimal_constant_flows.txt"))*0.9)
    upper_bounds = list(np.loadtxt(str("v" +version +"/optimal_constant_flows.txt"))*1.1)
    '''
    for i in range(11):
        lower_bounds.append(0.7)
        upper_bounds.append(4.0)
    '''
    lower_bounds.append(-1e-1) # tss feedback
    upper_bounds.append(-1e-3)
    lower_bounds.append(0.0) # efd gain
    upper_bounds.append(1.0)
        
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
    
elif evaluating == 'both' and mode == "optimize":
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


elif evaluating == "constant-flow" and mode == "compare":
    print("\nComparing optimization methods for constant flow in epsilon scenario\n")
    
    # Common setup for all methods
    lower_bounds = [0.7] * 11  # Eleven parameters for epsilon
    upper_bounds = [4.0] * 11
    search_space = Box(lower_bounds, upper_bounds)
    
    # Import required libraries for additional optimization methods
    from scipy.optimize import dual_annealing, differential_evolution
    import time
    from trieste.acquisition.function import ExpectedImprovement
    import matplotlib.ticker as mticker
    import csv
    
    # For tracking function calls
    function_call_count = 0
    
    # Define a combined objective function for methods that don't separate constraint and objective
    def combined_objective(params):
        global function_call_count
        function_call_count += 1
        
        params_array = np.array(params).flatten()
        
        # Use existing simulation infrastructure to evaluate
        data = run_swmm(params_array, None, verbose=False)
        return sum(data['data_log']['performance_measure'])
    
    # For tracking performance across methods
    results = {
        "BOUC": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf'), "pystorms_costs": []},
        "BO": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
        "DA": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')},
        "DE": {"costs": [], "fcalls": [], "best_params": None, "best_cost": float('inf')}
    }
    
    # Generate common initial points for all methods
    num_initial_points = 50  # Number of initial points
    num_steps = 200  # Number of optimization steps
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
    bouc_fcalls = 0
    obj_data = initial_data[OBJECTIVE]
    con_data = initial_data[CONSTRAINT]
    best_feasible_obj = float('inf')
    best_feasible_point = None
    bouc_costs = []
    bouc_fcalls_arr = []
    bouc_pystorms_costs = []

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
        
        # Calculate pystorms cost for this point
        data = run_swmm(point, None, verbose=False)
        pystorms_cost = sum(data['data_log']['performance_measure'])
        bouc_pystorms_costs.append(pystorms_cost)
        
        # Record original BOUC objective value
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
    results["BOUC"]["pystorms_costs"] = bouc_pystorms_costs
    results["BOUC"]["fcalls"] = bouc_fcalls_arr
    
    # Use the already calculated pystorms cost for the best point
    if best_feasible_point is not None:
        results["BOUC"]["best_params"] = best_feasible_point
        results["BOUC"]["best_cost"] = bouc_pystorms_costs[best_feasible_idx]
    else:
        # If no feasible point, use the best objective point
        idx = np.argmin(obj_data.observations)
        results["BOUC"]["best_params"] = obj_data.query_points[idx].numpy()
        results["BOUC"]["best_cost"] = bouc_pystorms_costs[idx]

    # Report correct function call counts
    print("BOUC function calls:", bouc_fcalls_arr[-1])
    print("BOUC best cost:", results["BOUC"]["best_cost"])
    
    # 2. Vanilla Bayesian Optimization using Trieste
    print("\nRunning vanilla BO optimization...")
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
    
    # Report correct function call counts
    print("BO function calls:", bo_fcalls_arr[-1])
    print("BO best cost:", results["BO"]["best_cost"])
    
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
    da_max_calls = num_steps * 2
    da_max_iter = num_steps // 5
    
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
    
    # Report correct function call counts
    print("DA function calls:", da_fcalls_arr[-1])
    print("DA best cost:", results["DA"]["best_cost"])
    
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
    print(f"DE max iterations: {de_max_iter}, popsize: {popsize}, function evaluations: {(de_max_iter + 1) * popsize * (len(lower_bounds) - 1)}")
    
    de_start_time = time.time()
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
    
    # Report correct function call counts
    print("DE function calls:", de_fcalls_arr[-1])
    print("DE best cost:", results["DE"]["best_cost"])
    
    # Compare results
    print("\nOptimization Results Comparison (using consistent performance measure):")
    print("Starting costs (from pystorms performance measure):")
    print(f"  BOUC: {bouc_pystorms_costs[0]:.4f}, BO: {bo_costs[0]:.4f}, DA: {da_costs[0]:.4f}, DE: {de_costs[0]:.4f}")
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

    # For BOUC, use pystorms cost directly
    results["BOUC"]["best_cost_so_far"] = rolling_min(results["BOUC"]["pystorms_costs"])

    # For other methods, costs already use pystorms cost
    for method in ["BO", "DA", "DE"]:
        results[method]["best_cost_so_far"] = rolling_min(results[method]["costs"])

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    
    # give the y axis a log10 scale
    plt.yscale('log')
    
    for method, data in results.items():
        if data["fcalls"] and data["best_cost_so_far"]:
            plt.plot(data["fcalls"], data["best_cost_so_far"], 'o-', label=labels[method], markersize=8)
            
    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('Best Cost Found', fontsize=14)
    plt.title('Optimization Methods Comparison by Function Evaluations - Epsilon Scenario', fontsize=16)
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
    plt.ylim(0.95 * min_cost_all, 2.0 * min_cost_all)

    for method, data in results.items():
        if data["fcalls"] and data["best_cost_so_far"]:
            plt.plot(data["fcalls"], data["best_cost_so_far"], 'o-', label=labels[method], markersize=8)

    plt.xlabel('Function Evaluations', fontsize=14)
    plt.ylabel('Best Cost Found', fontsize=14)
    plt.title('Optimization Methods Comparison (Zoomed) - Epsilon Scenario', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls_zoom.png")
    plt.savefig(f"v{version}/optimization_methods_comparison_by_fcalls_zoom.svg")
    plt.show()