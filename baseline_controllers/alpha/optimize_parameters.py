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
from skopt import gp_minimize, gbrt_minimize
from skopt.space import Real, Integer
import swmmio
import copy
import scipy
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


# ALPHA
evaluating = "both" # "constant-flow" or "efd" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
level = "1"
# level should always be 1 when optimizing parameters. controllers will be evaluated but not optimized on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))


# evaluating a given set of parameters for their cost
def run_swmm(constant_flows, efd_gain=None,verbose=False):
    env = pystorms.scenarios.alpha(version=version,level="1")

    env.env.sim.start()
    done = False

    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    #u_open_pct = constant_flows
    # make u_open_pct a deep copy of constant_flows (constant_flows should not change)
    u_open_pct = copy.deepcopy(constant_flows)

    states = pd.DataFrame(columns = env.config['states'])
    actions = pd.DataFrame(columns = env.config['action_space'])
    # indices of actions that are orifices (contain "Or")
    orifice_indices = [i for i in range(len(env.config['action_space'])) if "Or" in env.config['action_space'][i]]

    #print(env.env.actuator_schedule)
    #print(env.env.sensor_schedule)
    

    # grab some metadata for the control algorithm
    max_depths = dict()
    peak_filling_degrees = dict()
    for idx in range(1,6):
        regulator_id = "R" + str(idx)
        max_depths[regulator_id] = pyswmm.Nodes(env.env.sim)[regulator_id].full_depth
        peak_filling_degrees[regulator_id] = 0.0
    #print(max_depths)
    

    model = swmmio.Model(env.config["swmm_input"])
    #print(model.inp.orifices)
    orifice_areas = dict()
    for ori in model.inp.orifices.index.tolist():
        # Geom1 is the diameter of the orifice
        orifice_areas[ori] = np.pi*(model.inp.xsections.loc[ori, 'Geom1']/2)**2
        
    # per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
    # all orifices in alpha are "bottom"
    Cd = 0.65 # same for all 
    g = 32.2 # ft / s^2 (imperial units)
    # the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)

    while not done:
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state() # only the first eleven, controlled basins
            
            
            # iterate over the action space entries containing "Or" (orifices)
            for idx in orifice_indices:
                orifice_id = env.config['action_space'][idx]
                # the area of the orifice is the area of the circle
                Ao = orifice_areas[orifice_id]
                # Q = Cd * Ao * sqrt(2*g*H_e)
                Q_desired = constant_flows[idx] 
                # regulator depth
                # find the index of "R" + idx in the states
                regulator_id = "R" + orifice_id[2:]
                # find the index of the entry in the states which contains the regulator_id
                regulator_idx = [i for i in range(len(env.config['states'])) if regulator_id in env.config['states'][i]][0]
                    
                # assume linear scaling with opening percentage
                if state[regulator_idx] > 0:
                    u_open_pct[idx] = Q_desired / (Cd * Ao * np.sqrt(2*g*state[regulator_idx]) )
                if u_open_pct[idx] > 1.0:
                    u_open_pct[idx] = 1.0
                elif u_open_pct[idx] < 0.0:
                    u_open_pct[idx] = 0.0
                    
            filling_degrees = []
            for idx in orifice_indices: 
                orifice_id = env.config['action_space'][idx]
                regulator_id = "R" + orifice_id[2:]
                max_depth = max_depths[regulator_id]
                    
                # find the index of the entry in the states which contains the regulator_id
                regulator_idx = [i for i in range(len(env.config['states'])) if regulator_id in env.config['states'][i]][0]
                filling_degrees.append(state[regulator_idx] / max_depth)
               
                if state[regulator_idx] / max_depth > peak_filling_degrees[regulator_id]:
                    peak_filling_degrees[regulator_id] = state[regulator_idx] / max_depth
                
            filling_degrees = np.array(filling_degrees)
            filling_degree_avg = np.mean(filling_degrees)
            
            if evaluating == "constant-flow":
                
                done = env.step(u_open_pct.flatten())
            elif evaluating == "efd":
                for idx in orifice_indices:
                    this_fd = filling_degrees[idx]
                    u_open_pct[idx] += efd_gain * (this_fd - filling_degree_avg)

                for i in range(len(u_open_pct)):
                    if u_open_pct[i] > 1.0:
                        u_open_pct[i] = 1.0
                    elif u_open_pct[i] < 0.0:
                        u_open_pct[i] = 0.0
                done = env.step(u_open_pct.flatten())

            else:
                print("error. control scenario not recongized.")
                done = True
            
            if (not done) and verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 1 == 0: 
                u_print = u_open_pct.flatten()
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
            
            if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=2):
                final_depths = env.state()
                
        else:
            done = env.step(u_open_pct.flatten()) 
    
    return {"data_log": env.data_log, "final_depths": final_depths, "peak_filling_degrees": peak_filling_degrees}

    
#run_swmm(0.5*np.ones((10,1)), efd_parameters=[0.25], verbose=True)
'''

def f_constant_flows(constant_flows):
    # flatten the actions
    constant_flow_params = np.array(constant_flows).flatten()    

    data = run_swmm(constant_flow_params, None,verbose=False)
    CSO_flows = data['data_log']['flow']
    flooding = data['data_log']['flooding']
    CSO_penalty = 0
    for weir in CSO_flows.keys():
        CSO_penalty += sum(CSO_flows[weir])**2
    flooding_penalty = 0
    for junction in flooding.keys():
        #flooding_penalty += sum(flooding[junction])*100
        if sum(flooding[junction]) > 0:
            return 1e12

    # provide an incentive (negative cost) for getting the peak filling degrees close to 1.0
    # scale the reward quadratically by distance from 0.8
    peak_filling_degrees = data['peak_filling_degrees']
    peak_filling_reward = 0
    for regulator in peak_filling_degrees.keys():
        # give the maximum reward when about to flood
        peak_filling_reward += 1/((peak_filling_degrees[regulator] - 1.0)**6)


    return_value = CSO_penalty + flooding_penalty  + sum(data["final_depths"]) + np.std(data['final_depths'])
 
    #return_value = CSO_penalty + flooding_penalty - peak_filling_reward + sum(data["final_depths"]) + np.std(data['final_depths'])
    #print(return_value)
    return float(return_value)

def f_efd(efd_parameters):
    efd_params = np.array(efd_parameters).flatten()

    data = run_swmm(optimal_constant_flows, efd_params,verbose=False)
    CSO_flows = data['data_log']['flow']
    flooding = data['data_log']['flooding']
    CSO_penalty = 0
    for weir in CSO_flows.keys():
        CSO_penalty += sum(CSO_flows[weir])**2
    '''
    flooding_penalty = 0
    for junction in flooding.keys():
        flooding_penalty += sum(flooding[junction])*100

    # provide an incentive (negative cost) for getting the peak filling degrees close to 1.0
    # scale the reward quadratically by distance from 1.0
    peak_filling_degrees = data['peak_filling_degrees']
    peak_filling_reward = 0
    for regulator in peak_filling_degrees.keys():
        if peak_filling_degrees[regulator] > 0.95: # don't want to reward getting that close
            peak_filling_reward += 0
        else:
            peak_filling_reward += 10*(1/(1.0 - peak_filling_degrees[regulator]) )

    return_value = CSO_penalty + flooding_penalty - peak_filling_reward + sum(data["final_depths"]) + np.std(data['final_depths'])
    
    return return_value
    '''
    flooding_penalty = 0
    for junction in flooding.keys():
        #flooding_penalty += sum(flooding[junction])*100
        if sum(flooding[junction]) > 0:
            return 1e12

    # provide an incentive (negative cost) for getting the peak filling degrees close to 1.0
    # scale the reward quadratically by distance from 0.8
    peak_filling_degrees = data['peak_filling_degrees']
    peak_filling_reward = 0
    for regulator in peak_filling_degrees.keys():
        # give the maximum reward when about to flood
        peak_filling_reward += 1/((peak_filling_degrees[regulator] - 1.0)**6)


    return_value = CSO_penalty + flooding_penalty + sum(data["final_depths"]) + np.std(data['final_depths'])
    #return_value = CSO_penalty + flooding_penalty - peak_filling_reward + sum(data["final_depths"]) + np.std(data['final_depths'])
    #print(return_value)
    return float(return_value)
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
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_cost += sum(value)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
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
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_cost += sum(value)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
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
                efd_gain = np.array(sample).flatten()[-1]
                constant_flow_params = np.array(sample).flatten()[:-1]    
                data = run_swmm(constant_flow_params, efd_gain,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_cost += sum(value)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
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
                efd_gain = np.array(sample).flatten()[-1]
                constant_flow_params = np.array(sample).flatten()[:-1]     
                data = run_swmm(constant_flow_params, efd_gain,verbose=False)
                flow_cost = 0.0
                for key,value in data['data_log']['flow'].items():
                    flow_cost += sum(value)
                objective_cost = flow_cost + sum(data['final_depths']) + np.std(data['final_depths'])
                flood_cost = 0.0
                for key, value in data['data_log']['flooding'].items():
                    flood_cost += sum(value)
                # if flood cost is more than zero, ensure it's more than one
                if flood_cost > 0.0 and flood_cost < 1.0:
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
    for i in range(10):
        if i < 5: # orifices, set constant flow rate
            lower_bounds.append(0.5)
            upper_bounds.append(10.0)
        else: # regulators, set constant opening percentage
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 20
    initial_data = observer_cf(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 20
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
    for i in range(10):
        if i < 5: # orifices, set constant flow rate
            lower_bounds.append(0.5)
            upper_bounds.append(10.0)
        else: # regulators, set constant opening percentage
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
    # efd gain bounds
    lower_bounds.append(0.0)
    upper_bounds.append(2.0)
        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 20
    initial_data = observer_efd(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_efd.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 20
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
    
elif evaluating == "both":
    evaluating = "constant-flow"
    lower_bounds = []
    upper_bounds = []
    for i in range(10):
        if i < 5: # orifices, set constant flow rate
            lower_bounds.append(0.5)
            upper_bounds.append(10.0)
        else: # regulators, set constant opening percentage
            lower_bounds.append(0.0)
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

    num_steps = 1000
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
    
    evaluating = "efd"
    lower_bounds = []
    upper_bounds = []
    for i in range(10):
        if i < 5: # orifices, set constant flow rate
            lower_bounds.append(0.5)
            upper_bounds.append(10.0)
        else: # regulators, set constant opening percentage
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
    # efd gain bounds
    lower_bounds.append(0.0)
    upper_bounds.append(2.0)
        
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 110
    initial_data = observer_efd(search_space.sample(num_initial_points))
    
    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_efd.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 1200
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