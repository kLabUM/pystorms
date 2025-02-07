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

# THETA
evaluating = "both" # "constant-flow" or "efd" or "both"
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
                u_open_pct = u_open_pct + efd_parameters[0]*u_diff
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


def feas_constant_flows(constant_flows):
    constant_flow_params = np.array(constant_flows).flatten()    

    # defines the feasible region. anwywhere flooding occurs is infeasible, anywhere else is fine.

    data = run_swmm(constant_flow_params, None,verbose=False)
    if any(any(data['data_log']['flooding'])) > 0.0:
        return 1.0
    else:
        return 0.0

def f_constant_flows(constant_flows):
    # flatten the actions
    constant_flow_params = np.array(constant_flows).flatten()    

    data = run_swmm(constant_flow_params, None,verbose=False)
    # don't penalize flooding as it's defining the feasible region. want cost to slope downward toward that boundary.


    return_value = sum(data['data_log']['performance_measure']) + sum(data["final_depths"]) + 10*np.std(data['final_depths'])
    return float(return_value)

def f_efd(efd_parameters):
    efd_params = np.array(efd_parameters).flatten()

    data = run_swmm(optimal_constant_flows, efd_params,verbose=False)
    return float(data['cost'] + sum(data['final_depths'])) + 10*np.std(data['final_depths'])

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
                    flow_exceed = [x - 0.5 for x in value]
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
                    flow_exceed = [x - 0.5 for x in value]
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
                    
                Sim_cf.computed_return_values[tuple(sample.numpy())] = dict()
                Sim_cf.computed_return_values[tuple(sample.numpy())]['objective'] = objective_cost
                Sim_cf.computed_return_values[tuple(sample.numpy())]['constraint'] = flood_cost
                return_values.append(flood_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    


    '''
    @staticmethod
    def objective(input_data):
        return_values = []
        for sample in input_data:
            constant_flow_params = np.array(sample).flatten()    
            data = run_swmm(constant_flow_params, None,verbose=False)
            flow_cost = 0.0
            for key,value in data['data_log']['flow'].items():
                flow_exceed = [x - 0.5 for x in value]
                flow_exceed = [x if x > 0 else 0 for x in flow_exceed]
                flow_cost += 10*sum(flow_exceed)
            return_values.append(flow_cost + sum(data['final_depths']) + 10*np.std(data['final_depths']))
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
    
    @staticmethod
    def constraint(input_data):
        return_values = []
        for sample in input_data:
            constant_flow_params = np.array(sample).flatten()    
            data = run_swmm(constant_flow_params, None,verbose=False)
            flood_cost = 0.0
            for key, value in data['data_log']['flooding'].items():
                flood_cost += 10*sum(value)
            return_values.append(flood_cost)
        return_values = np.array(return_values).reshape(-1,1)
        return return_values
   '''
OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"
    
def observer(query_points):
    return {
            OBJECTIVE: Dataset(query_points, Sim_cf.objective(query_points)),
            CONSTRAINT: Dataset(query_points, Sim_cf.constraint(query_points)),
        }

if evaluating == "constant-flow":
    domain = []
    x0 = []
    for i in range(1, 3):
        domain.append(Real(0.05,0.5,name=str(i)))
        x0.append(0.24)
    
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=100, n_initial_points=20, 
                     initial_point_generator = 'lhs',verbose=True)
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal constant heads 
    np.savetxt(str("v" +version +"/optimal_constant_flows.txt"), bo.x)


elif evaluating == "efd":
    optimal_constant_flows = np.loadtxt(str(version +"/optimal_constant_flows.txt"))
    #domain = [{"name": "TSS_feedback", "type": "continuous", "domain": (-1, -1e-4)},
    #        {"name": "efd_gain", "type": "continuous", "domain": (0.0, 1.0)}]
    domain = [Real(0.0, 1.0, name="efd_gain")]
    x0 = [0.5]
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=25, n_initial_points=2, 
                     initial_point_generator = 'lhs',verbose=True)
    
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" +version +"/optimal_efd_params.txt"), bo.x)    

elif evaluating == 'both':
    evaluating="constant-flow"
    domain = []
    x0 = []
    
    lower_bounds = []
    upper_bounds = []
    for i in range(1, 3):
        domain.append(Real(0.01,0.5,name=str(i)))
        x0.append(0.1)
        lower_bounds.append(0.01)
        upper_bounds.append(0.5)
    search_space = Box(lower_bounds, upper_bounds)
    
    num_initial_points = 10
    initial_data = observer(search_space.sample(num_initial_points))
    
    from trieste.models.gpflow import build_gpr, GaussianProcessRegression

    # likelihood variacne very small as model is assumed to be noiseless
    def create_bo_model(data):
        gpr = build_gpr(data, search_space)
        return GaussianProcessRegression(gpr)


    initial_models = trieste.utils.map_values(create_bo_model, initial_data)

    from trieste.acquisition.rule import EfficientGlobalOptimization

    pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim_cf.threshold)
    eci = trieste.acquisition.ExpectedConstrainedImprovement(
        OBJECTIVE, pof.using(CONSTRAINT)
    )
    rule = EfficientGlobalOptimization(eci)  # type: ignore

    num_steps = 50
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

    opt_result = bo.optimize(
        num_steps, initial_data, initial_models, rule)
    data = opt_result.try_get_final_datasets()
    models = opt_result.try_get_final_models()
    
    constraint_data = data[CONSTRAINT]
    #new_query_points = constraint_data.query_points[-num_steps:]
    #new_observations = constraint_data.observations[-num_steps:]
    #new_data = (new_query_points, new_observations)

    # plot the model and observations of the objective function
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    # query points is 2 dimensional, so make a heatmap with observations as the z variable
    # locations will be: data['OBJECITVE'].query_points
    # z values will be: data['OBJECTIVE'].observations
    # plot the observations
    ax[0].scatter(data[OBJECTIVE].query_points[:, 0], data[OBJECTIVE].query_points[:,1], c=data[OBJECTIVE].observations, label="observations")


    # plot the GP model outputs across a grid of points
    # make a grid of two-dimensional points across the search space
    x1 = np.linspace(0.01, 0.5, 100)
    x2 = np.linspace(0.01, 0.5, 100)
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
    # add a black contour line at 0.0 to show the boundary of the feasible region
    ax[1].contour(X1, X2, constraint_mean_predicted.reshape(100,100), levels=[0.0], colors='black')
    plt.colorbar(c, ax=ax[1])
    ax[1].set_title("Constraint observations and gp model")
    ax[1].set_xlabel("Constant Flow 1")
    plt.savefig(str("v" +version + "/constrained_bo.png"))
    plt.savefig(str("v" +version + "/constrained_bo.svg"))
    plt.show(block=True)
    

    



    #bo = gp_minimize(f_constant_flows, domain,x0=x0, 
    #                 n_calls=250, n_initial_points=25, 
    #                 initial_point_generator = 'lhs',verbose=True)
    #print(bo.x)
    #print(bo.fun)
    #print(bo.x_iters)
    
    # save the optimal constant heads 
    np.savetxt(str("v" +version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" +version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))

    # save the whole bo object
    #with open("bo_constant_flows.pkl", "wb") as f:
    #    pickle.dump(bo, f)


    evaluating = "efd"
    optimal_constant_flows = np.loadtxt(str("v" +version +"/optimal_constant_flows.txt"))
    #domain = [{"name": "TSS_feedback", "type": "continuous", "domain": (-1, -1e-4)},
    #        {"name": "efd_gain", "type": "continuous", "domain": (0.0, 1.0)}]
    domain = [Real(0.0, 1.0, name="efd_gain")]
    x0 = [0.01]
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=100, n_initial_points=25, 
                     initial_point_generator = 'lhs',verbose=True)
    
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" +version + "/optimal_efd_params.txt"), bo.x)
    # save bo.fun
    np.savetxt(str("v" +version + "/optimal_efd_cost.txt"), np.array([bo.fun]))
    # save the whole bo object
    #with open("bo_efd.pkl", "wb") as f:
    #    pickle.dump(bo, f)
    
    