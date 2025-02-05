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

# GAMMA
evaluating = "both" # "constant-flow" or "efd" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
level = "1"
# level should always be 1 when optimizing parameters. controllers will be evaluated but not optimized on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))


# evaluating a given set of parameters for their cost
def run_swmm(constant_flows, efd_parameters=None,verbose=False):
    
    env = pystorms.scenarios.gamma()
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    # the code here removing 5 and 9 from the scenario will live in gamma.py eventually
    for state in env.config['states']:
        # if state contains 5 or 9, remove it
        if '5' in state[0] or '9' in state[0]:
            env.config['states'].remove(state)
    for action in env.config['action_space']:
        # if action contains 5 or 9, remove it
        if '5' in action or '9' in action:
            env.config['action_space'].remove(action)
    for target in env.config['performance_targets']:
        # if target contains 5 or 9, remove it
        if '5' in target[0] or '9' in target[0]:
            env.config['performance_targets'].remove(target)
    #print(env.config['states'])
    #print(env.config['action_space'])
    #print(env.config['performance_targets'])
    env.env.sim.start()
    done = False
    last_eval = env.env.sim.start_time - datetime.timedelta(hours=1) 
    last_read = env.env.sim.start_time - datetime.timedelta(hours=1)
    start_time = env.env.sim.start_time
    
    u_open_pct = 0.0*np.ones(len(env.config['action_space'])) # start closed

    states = pd.DataFrame(columns = env.config['states'])
    actions = pd.DataFrame(columns = env.config['action_space'])

    #print(env.env.actuator_schedule)
    #print(env.env.sensor_schedule)

    max_depths_array = np.array([]) # for all states
    for state in env.config['states']:
        if 'depthN' in state[1]:
            node_id = state[0]
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)


    model = swmmio.Model(env.config["swmm_input"])
    #print(model.inp.orifices)
    orifice_areas = dict()
    for ori in model.inp.orifices.index.tolist():
        # Geom1 and geom2 are the height and width of the orifice
        orifice_areas[ori] = model.inp.xsections.loc[ori, 'Geom1']*model.inp.xsections.loc[ori, 'Geom2']
        
    # per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
    # all orifices in gamma are "bottom" 
    Cd = 0.65 # same for all 
    g = 32.2 # ft / s^2 (imperial units)
    # the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)

    
    while not done:
        
        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time
            state = env.state(level=level)
            
            # iterate over the action space entries containing "Or" (orifices)
            for idx in range(len(env.config['action_space'])):
                orifice_id = env.config['action_space'][idx]
                # the area of the orifice is the area of the circle
                Ao = orifice_areas[orifice_id]
                # Q = Cd * Ao * sqrt(2*g*H_e)
                Q_desired = constant_flows[idx] 

                # assume linear scaling with opening percentage
                if state[idx] > 0.0:
                    u_open_pct[idx] = Q_desired / (Cd * Ao * np.sqrt(2*g*state[idx]))
                else:
                    u_open_pct[idx] = 1.0
                
                if u_open_pct[idx] > 1.0:
                    u_open_pct[idx] = 1.0
                elif u_open_pct[idx] < 0.0:
                    u_open_pct[idx] = 0.0
            
            filling_degrees = state / max_depths_array
            filling_degree_avg = np.mean(filling_degrees)
            
            if evaluating == "constant-flow":

                done = env.step(u_open_pct.flatten())
            elif evaluating == "equal-filling":
                for idx in range(len(env.config['action_space'])):
                    this_fd = filling_degrees[idx]
                    u_open_pct[idx] += efd_parameters[0] * (this_fd - filling_degree_avg)

                for i in range(len(u_open_pct)):
                    if u_open_pct[i] > 1.0:
                        u_open_pct[i] = 1.0
                    elif u_open_pct[i] < 0.0:
                        u_open_pct[i] = 0.0
                done = env.step(u_open_pct.flatten())

            else:
                print("error. control scenario not recongized.")
                done = True
        if (not done) and verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 4 == 0: 
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
            
        #done = env.step(u_open_pct.flatten(),level=level)
        done = env.step(u_open_pct.flatten())
        
    return {"data_log": env.data_log}

#run_swmm(np.ones(9), None, verbose=True)

def f_constant_flows(constant_flows):
    data = run_swmm(constant_flows, None, verbose=False)
    # rather than penalize on the threshold 0.1, penalize the final storage depth for a smoother gradient
    flow_penalty = 0.0
    for key, value in data['data_log']['flow'].items():
        # if the key doesn't contain 5 or 9
        if '5' not in key and '9' not in key:
            value = [x - 4.0 for x in value]
            value = [x if x > 0.0 else 0.0 for x in value]
            flow_penalty += sum(value)
    drainage_penalty = 0.0
    for key, value in data['data_log']['depthN'].items():
        if '5' not in key and '9' not in key:
            final_depth = value[-1]
            drainage_penalty += 1000*final_depth
        
    return flow_penalty + drainage_penalty

def f_efd(efd_parameters):
    data = run_swmm(optimal_constant_flows, efd_parameters, verbose=False)
    # rather than penalize on the threshold 0.1, penalize the final storage depth for a smoother gradient
    flow_penalty = 0.0
    for key, value in data['data_log']['flow'].items():
        # if the key doesn't contain 5 or 9
        if '5' not in key and '9' not in key:
            value = [x - 4.0 for x in value]
            value = [x if x > 0.0 else 0.0 for x in value]
            flow_penalty += sum(value)
    drainage_penalty = 0.0
    for key, value in data['data_log']['depthN'].items():
        if '5' not in key and '9' not in key:
            final_depth = value[-1]
            drainage_penalty += 1000*final_depth
        
    return flow_penalty + drainage_penalty

if evaluating == "constant-flow":
    domain = []
    x0 = []
    for i in range(9):
        domain.append(Real(3.0,10.0))
        x0.append(4.5)
        

    
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=10, n_initial_points=5, 
                     initial_point_generator = 'lhs',verbose=True)
    #bo = gp_minimize(f_constant_flows, domain,x0=x0, verbose=True)
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal constant flows
    np.savetxt(str("v" + version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" + version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))
    

elif evaluating == "equal-filling":
    optimal_constant_flows = np.loadtxt(str("v" + version +"/optimal_constant_flows.txt"))
    #domain = [{"name": "TSS_feedback", "type": "continuous", "domain": (-1, -1e-4)},
    #        {"name": "efd_gain", "type": "continuous", "domain": (0.0, 1.0)}]
    domain = [Real(0.0, 10.0, name="efd_gain")]
    x0 = [0.5] 
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=10, n_initial_points=5, 
                        initial_point_generator = 'lhs',verbose=True)
    
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" + version +"/optimal_efd_params.txt"), bo.x)
    # save bo.fun
    np.savetxt(str("v" + version +"/optimal_efd_cost.txt"), np.array([bo.fun]))
    
elif evaluating == "both":
    evaluating = "constant-flow"
    domain = []
    x0 = []
    for i in range(9):
        domain.append(Real(3.0,10.0))
        x0.append(4.5)
        

    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=500, n_initial_points=50, 
                     initial_point_generator = 'lhs',verbose=True)
    #bo = gp_minimize(f_constant_flows, domain,x0=x0, verbose=True)
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal constant flows
    np.savetxt(str("v" + version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" + version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))
    
    evaluating="equal-filling"
    optimal_constant_flows = np.loadtxt(str("v" + version +"/optimal_constant_flows.txt"))
    #domain = [{"name": "TSS_feedback", "type": "continuous", "domain": (-1, -1e-4)},
    #        {"name": "efd_gain", "type": "continuous", "domain": (0.0, 1.0)}]
    domain = [Real(1e-5, 10.0, name="efd_gain")]
    x0 = [0.1] 
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=250, n_initial_points=25, 
                        initial_point_generator = 'lhs',verbose=True)
    
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" + version +"/optimal_efd_params.txt"), bo.x)
    # save bo.fun
    np.savetxt(str("v" + version +"/optimal_efd_cost.txt"), np.array([bo.fun]))
