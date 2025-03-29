'''
# install pystorms from the current directory (this should be commented out in final version once pystorms source code isn't changing all the time)
import subprocess
import sys
from tkinter import N
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pystorms'])
subprocess.check_call([sys.executable, '-m', 'pip', 'cache', 'purge'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
'''
import pystorms # this will be the first line of the program when dev is done
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyswmm
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
import swmmio
import copy
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
import dill as pickle
# DELTA
evaluating = "both" # "static-plus-rule" or "prop-outflow" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
level = "1"
# level should always be 1 when optimizing parameters. controllers will be evaluated but not optimized on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))
    
operational_bounds_data = {
    "Lower Limit": [5.32, 4.44, 5.20, 3.28, None],
    "Upper Limit": [5.92, 5.04, 5.80, 3.80, 6.55]
}

operational_bounds_df = pd.DataFrame(operational_bounds_data, index=["N3", "N2", "N1", "C", "S"])

exceedance_bounds_data = {
    "Lower Limit": [5.28, 4.04, 2.11, 2.21, None],
    "Upper Limit": [11.99, 6.59, 5.92, 5.70, 9.55]
}

exceedance_bounds_df = pd.DataFrame(exceedance_bounds_data, index=["N3", "N2", "N1", "C", "S"])

def run_swmm(static_settings, prop_gain=None, verbose=False):
    env = pystorms.scenarios.delta(version=version,level=level)

    env.env.sim.start()
    done = False
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    #u_open_pct = constant_flows
    # make u_open_pct a deep copy of constant_flows (constant_flows should not change)
    #u_open_pct = copy.deepcopy(static_settings)
    u_open_pct = copy.deepcopy(static_settings[:-1])

    states = pd.DataFrame(columns = env.config['states'])
    actions = pd.DataFrame(columns = env.config['action_space'])
    max_depth = dict()
    for key in env.config['states']:
        max_depth[key[0]] = pyswmm.Nodes(env.env.sim)[key[0]].full_depth
    
    while not done:
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state(level=level)
            
            # first bit is the same for all 3 controllers
            # set the weirs
            #u_open_pct[:-1] = static_settings[:-1]
            u_open_pct[:-1] = static_settings[:-2]
            # open valve?
            if evaluating != "uncontrolled":
                '''
                if state[0] > static_settings[-1]: # basin c depth above optimized threshold?
                    u_open_pct[-1] = 1.0 # fully open valve above threshold depth in basin c
                else:
                    u_open_pct[-1] = 0.0 # close the valve to preserve capacity in the infiltration basin    
                '''
                # try proprtional valve opening between two setpoint depths
                if state[0] < static_settings[-2]: # below lower threshold
                    u_open_pct[-1] = 0.0 # close the valve to preserve capacity in the infiltration basin
                elif state[0] > static_settings[-1]: # above upper threshold
                    u_open_pct[-1] = 1.0 # fully open valve above upper threshold depth in basin c
                else: # between the two thresholds
                    # linearly interpolate the valve opening based on the current depth in basin C
                    # find the range between the two thresholds
                    lower_threshold = static_settings[-2] # lower threshold depth
                    upper_threshold = static_settings[-1] # upper threshold depth
                    range_threshold = upper_threshold - lower_threshold
                    # calculate the current depth in the range
                    current_depth_in_range = state[0] - lower_threshold
                    # calculate the percentage of the way through the range
                    percentage_in_range = current_depth_in_range / range_threshold
                    # set the valve opening based on the percentage in range
                    u_open_pct[-1] = percentage_in_range # linearly interpolate the valve opening based on the current depth in basin C
                            

            if evaluating == "prop-outflow":
                # how much are outflows below the threshold?
                current_outflow = pyswmm.Links(env.env.sim)['conduit_Edown'].flow
                flow_capacity = env.threshold - current_outflow
                # flow capacity may be negative, in which case the weirs would raise to reduce outflows

                for idx, name_w_weir_prefix in enumerate(env.config['action_space'][:-1]):
                    name = name_w_weir_prefix[5:]
                    # find the index of the entry in "states" which contains that name. it will contain it but not be an exact match
                    idx_state = [i for i, s in enumerate(states.columns) if name in s[0]]
                    op_bound_width = operational_bounds_df.loc[name,"Upper Limit"] - operational_bounds_df.loc[name, "Lower Limit"]
                    lower_tight_bound = operational_bounds_df.loc[name, "Lower Limit"] + 0.1 * op_bound_width
                    upper_tight_bound = operational_bounds_df.loc[name,"Upper Limit"] - 0.25 * op_bound_width
                    if state[idx_state[0]] < upper_tight_bound and state[idx_state[0]] > lower_tight_bound:
                        # only use proportional feedback on the outflow if we're safe regarding depth bounds
                        u_open_pct[idx] += prop_gain * flow_capacity
                    elif state[idx_state[0]] >= upper_tight_bound and prop_gain*flow_capacity > 0: # allow extra opening if close to exceeding upper bound
                        u_open_pct[idx] += prop_gain * flow_capacity
                    if u_open_pct[idx] > 1.0:
                        u_open_pct[idx] = 1.0
                    elif u_open_pct[idx] < 0.0:
                        u_open_pct[idx] = 0.0
                    

        if (not done) and verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 4 == 0: 
                u_print = u_open_pct
                y_measured = env.state().reshape(-1,1)
                # print the names of the states and their current values side-by-side
                print("state, value")
                for idx in range(len(env.config['states'])):
                    print(env.config['states'][idx], y_measured[idx])
                print("actuator, setting")
                for idx in range(len(env.config['action_space'])):
                    print(env.config['action_space'][idx], u_print[idx])

                print("current time, end time")
                print(env.env.sim.current_time, env.env.sim.end_time)
                print("\n")


        if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
            node_indices = [i for i in range(len(env.config['states'])) if 'depthN' in env.config['states'][i][1]]
            final_depths = env.state()[node_indices]
    
        done = env.step(u_open_pct,level=level)
        
    perf = sum(env.data_log["performance_measure"])
    print("performance of: ", evaluating)
    print("{:.4e}".format(perf))
    flood_cost = 0
    for key,value in env.data_log['flooding'].items():
        if sum(value) > 0:
            # the total cost is the number of positive entries in value multiplied by the cost of flooding (10e6)
            # find the number of positive entries in value
            num_positive = sum([1 for x in value if x > 0])
            print(key,"{:.4e}".format(num_positive*(10**6)))
            flood_cost += num_positive*(10**6)
    print("total flood cost:","{:.4e}".format(flood_cost))
    excess_flows = [x - env.threshold for x in env.data_log['flow']['conduit_Eout']] 
    excess_flows = [x if x > 0 else 0 for x in excess_flows]
    flow_cost = sum(excess_flows)*10
    print("total flow cost (leaving network exceeds threshold):","{:.4e}".format(flow_cost))
    #print("depth bounds cost:","{:.4e}".format(perf - flood_cost - flow_cost))
    op_bounds_cost = 0.0
    for key,value in env.data_log['depthN'].items():
        exceed_upper = [x - operational_bounds_df.loc[key[6:],"Upper Limit"] for x in value]
        exceed_upper = [x if x > 0 else 0 for x in exceed_upper]
        exceed_lower = [x - operational_bounds_df.loc[key[6:],"Lower Limit"] for x in value]
        exceed_lower = [x if x < 0 else 0 for x in exceed_lower]
        op_bounds_cost += np.abs(sum(exceed_upper)) + np.abs(sum(exceed_lower))
    exceedance_cost = 0.0
    for key,value in env.data_log['depthN'].items():
        exceed_upper = [x - exceedance_bounds_df.loc[key[6:],"Upper Limit"] for x in value]
        exceed_upper = [x if x > 0 else 0 for x in exceed_upper]
        exceed_lower = [x - exceedance_bounds_df.loc[key[6:],"Lower Limit"] for x in value]
        exceed_lower = [x if x < 0 else 0 for x in exceed_lower]
        exceedance_cost += np.abs(sum(exceed_upper)) + np.abs(sum(exceed_lower))
    # flood cost and exceedance cost will define feasibility
    # if exceedance cost is zero, record 1 - (closest approach to exceedance bounds)/(closest intitial distance from exceedance bounds)
    # if we never get closer to the exceedance bounds than our initial depths, then this records 0
    # otherwise it scales up to 1.0 as we barely avoid exceeding the bounds, normalized by our initial distance
    if exceedance_cost == 0.0:
        closest_initial_distance = np.inf
        closest_approach = np.inf
        for key,value in env.data_log['depthN'].items():
            initial_distance = np.abs(value[0] - exceedance_bounds_df.loc[key[6:],"Lower Limit"])
            if initial_distance < closest_initial_distance:
                closest_initial_distance = initial_distance
            initial_distance = np.abs(exceedance_bounds_df.loc[key[6:], "Upper Limit"] - value[0])
            if initial_distance < closest_initial_distance:
                closest_initial_distance = initial_distance

            approach = np.abs(np.array(value) - exceedance_bounds_df.loc[key[6:],"Lower Limit"])
            if min(approach) < closest_approach:
                closest_approach = min(approach)
            approach = np.abs(np.array(value) - exceedance_bounds_df.loc[key[6:],"Upper Limit"])
            if min(approach) < closest_approach:
                closest_approach = min(approach)
        exceedance_cost = 1 - (closest_approach/closest_initial_distance)
    print("operational bounds cost:","{:.4e}".format(op_bounds_cost))
    print("exceedance cost:","{:.4e}".format(exceedance_cost))

    # if exceedance cost is still zero, then report the maximum filling degree across all the recorded depths
    if exceedance_cost == 0.0 and flood_cost == 0.0:
        max_fill = 0
        for key,value in env.data_log['depthN'].items():
            fill = np.array(max(value))/max_depth[key]
            if fill > max_fill:
                max_fill = fill
        exceedance_cost = max_fill
    # round final_depths to 2 decimal places
    final_depths = np.round(final_depths,2)

    return {"data_log":env.data_log,"flow_cost":flow_cost,"op_bounds_cost":op_bounds_cost,"exceedance_cost":exceedance_cost,"flood_cost":flood_cost}


class Sim_spr:
    threshold = 0.99 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()
    
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_spr.computed_return_values.keys():
                return_values.append(Sim_spr.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                static_settings = np.array(sample).flatten()    
                data = run_swmm(static_settings, None,verbose=False)
                objective_cost = data['flow_cost'] + data['op_bounds_cost']
                constraint_cost = data['flood_cost'] + data['exceedance_cost']

                Sim_spr.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_spr.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_spr.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_cost
                return_values.append(objective_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_spr.computed_return_values.keys():
                return_values.append(Sim_spr.computed_return_values[tuple(sample.numpy())]['constraint'])
            else:
                static_settings = np.array(sample).flatten()    
                data = run_swmm(static_settings, None,verbose=False)
                objective_cost = data['flow_cost'] + data['op_bounds_cost']
                constraint_cost = data['flood_cost'] + data['exceedance_cost']

                
                Sim_spr.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_spr.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_spr.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_cost
                return_values.append(constraint_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
    
def observer_spr(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_spr.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_spr.constraint(query_points)),
        }

class Sim_po:
    threshold = 0.99 # on the constraint function to define the feasible (safe) region
    computed_return_values = dict()
    
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_po.computed_return_values.keys():
                return_values.append(Sim_po.computed_return_values[tuple(sample.numpy())]['objective'])
            else:
                static_settings = np.array(sample).flatten()[:-1] 
                prop_gain = sample[-1]
                data = run_swmm(static_settings, prop_gain,verbose=False)
                objective_cost = data['flow_cost'] + data['op_bounds_cost']
                constraint_cost = data['flood_cost'] + data['exceedance_cost']

                Sim_po.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_po.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_po.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_cost
                return_values.append(objective_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            if tuple(sample.numpy()) in Sim_po.computed_return_values.keys():
                return_values.append(Sim_po.computed_return_values[tuple(sample.numpy())]['constraint'])
            else:
                static_settings = np.array(sample).flatten()[:-1] 
                prop_gain = sample[-1]
                data = run_swmm(static_settings, prop_gain,verbose=False)
                objective_cost = data['flow_cost'] + data['op_bounds_cost']
                constraint_cost = data['flood_cost'] + data['exceedance_cost']

                
                Sim_po.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_po.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_po.computed_return_values[tuple(sample.numpy())]['constraint'] = constraint_cost
                return_values.append(constraint_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
def observer_po(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_po.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_po.constraint(query_points)),
        }



def create_bo_model(data):
        gpr = build_gpr(data, search_space)
        return GaussianProcessRegression(gpr)

if evaluating == "static-plus-rule":
    lower_bounds = [0.95, 0.33,0.10, 0.51, 3.0]
    upper_bounds = [1.0, 0.43, 0.2, 0.61, 4.0]

    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 2
    initial_data = observer_spr(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_spr.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 1
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_spr, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_spr.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_static.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_static_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        with open("bo_static-plus-rule.pkl", "wb") as f:
            pickle.dump(opt_result, f)
            

if evaluating == "prop-outflow":
    lower_bounds = [0.95, 0.33,0.10, 0.51, 3.0,1e-4]
    upper_bounds = [1.0, 0.43, 0.2, 0.61, 4.0,0.1]

    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 2
    initial_data = observer_po(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_po.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 1
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_po, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_po.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_prop.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_prop_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        with open("bo_prop-outflow.pkl", "wb") as f:
            pickle.dump(opt_result, f)
            
if evaluating == "both":
    evaluating = "static-plus-rule"
    lower_bounds = [0.95, 0.25,0.10, 0.4, 3.65 , 3.83]
    upper_bounds = [1.0, 0.38, 0.2, 0.55, 3.77 , 3.95]

    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 75
    initial_data = observer_spr(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_spr.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 200
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_spr, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_spr.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_static.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_static_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        with open("bo_static-plus-rule.pkl", "wb") as f:
            pickle.dump(opt_result, f)
    
    evaluating = "prop-outflow"
    #lower_bounds = [0.95, 0.25,0.10, 0.4, 3.0,1e-4]
    #upper_bounds = [1.0, 0.38, 0.2, 0.55, 4.0,0.1]
    lower_bounds = [0.95, 0.25,0.10, 0.4, 3.65 , 3.83,1e-4]
    upper_bounds = [1.0, 0.38, 0.2, 0.55, 3.77 , 3.95,0.1]

    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 100
    initial_data = observer_po(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_po.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 250
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer_po, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    # find the indices of the feasible query points
    feasible_indices = np.where(data[CONSTRAINT].observations <= Sim_po.threshold)[0]
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
        np.savetxt(str("v" +version +"/optimal_prop.txt"), best_feasible_point.numpy())
        np.savetxt(str("v" +version +"/optimal_prop_cost.txt"), best_feasible_observation.numpy())
        # save the whole object
        with open("bo_prop-outflow.pkl", "wb") as f:
            pickle.dump(opt_result, f)