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
# print current working directory
import os
print(os.getcwd())
# set the working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# ALPHA
evaluating = "both" # "constant-flow" or "efd" or "both"
version = "2" # "1" or "2" - 2 will be the updated, more difficult version
level = "1"
# level should always be 1 when optimizing parameters. controllers will be evaluated but not optimized on higher levels
# if the directory version doesn't exist, create it
if not os.path.exists(str("v"+version)):
    os.makedirs(str("v"+version))


# evaluating a given set of parameters for their cost
def run_swmm(constant_flows, efd_parameters=None,verbose=False):
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
            for idx in orifice_indices: # this is not correct right now
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
            elif evaluating == "equal-filling":
                for idx in orifice_indices:
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
            
            if (not done) and env.env.sim.current_time > env.env.sim.end_time - datetime.timedelta(hours=1):
                final_depths = env.state()
                
        else:
            done = env.step(u_open_pct.flatten()) 
    
    return {"data_log": env.data_log, "final_depths": final_depths, "peak_filling_degrees": peak_filling_degrees}

    
#run_swmm(0.5*np.ones((10,1)), efd_parameters=[0.25], verbose=True)

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


if evaluating == "constant-flow":
    domain = []
    x0 = []
    for i in range(10):
        if i < 5: # orifices, constant flow rates
            domain.append(Real(0.5,5.0))
            x0.append(3.0)
        else: # regulators, constant opening percentage
            domain.append(Real(0.0,1.0))
            x0.append(0.5)

    
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=10, n_initial_points=5, 
                     initial_point_generator = 'lhs',verbose=True, acq_func="LCB")
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
    domain = [Real(1e-5, 10.0, name="efd_gain")]
    x0 = [2.0] 
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=100, n_initial_points=10, 
                     initial_point_generator = 'lhs',verbose=True, acq_func="LCB",kappa=1.96)
    
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" + version +"/optimal_efd_params.txt"), bo.x)
    # save bo.fun
    np.savetxt(str("v" + version +"/optimal_efd_cost.txt"), np.array([bo.fun]))
    # save the whole bo object
    #with open("bo_efd.pkl", "wb") as f:
    #    pickle.dump(bo, f)
    
elif evaluating == "both":
    evaluating = "constant-flow"
    domain = []
    x0 = []
    for i in range(10):
        if i < 5: # orifices, constant flow rates
            domain.append(Real(0.5,10.0))
            x0.append(3.0)
        else: # regulators, constant opening percentage
            domain.append(Real(0.0,1.0))
            x0.append(0.5)
    #x0[0] = 3.0
    x0[5] = 0.2
    x0[7] = 0.45
    
    #opt_result = scipy.optimize.minimize(f_constant_flows, x0, jac = '3-point',
    #                                     method='trust-constr', 
    #                                     options={"xtol":1e-5,"verbose":3,disp":True,"initial_tr_radius":1e-2})
    
    #opt_result = scipy.optimize.minimize(f_constant_flows,x0, method='Nelder-Mead', 
    #                                     options={"fatol":1.0,"disp":True,"adaptive":True})
    #print(opt_result)
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=200, n_initial_points=25, 
                     initial_point_generator = 'lhs',verbose=True, acq_func="LCB",kappa=1e2)
    #bo = gp_minimize(f_constant_flows, domain,x0=x0, verbose=True)
    print(bo.x)
    print(bo.fun)
    #print(bo.x_iters)
    # save the optimal constant flows
    np.savetxt(str("v" + version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" + version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))
    
    evaluating = "equal-filling"
    optimal_constant_flows = np.loadtxt(str("v" + version +"/optimal_constant_flows.txt"))

    domain = [Real(1e-5, 10.0, name="efd_gain")]
    x0 = [2.0] 
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=100, n_initial_points=10, 
                     initial_point_generator = 'lhs',verbose=True, acq_func="LCB",kappa=1e2)
    
    print(bo.x)
    print(bo.fun)
    #print(bo.x_iters)
    # save the optimal efd params
    np.savetxt(str("v" + version +"/optimal_efd_params.txt"), bo.x)
    # save bo.fun
    np.savetxt(str("v" + version +"/optimal_efd_cost.txt"), np.array([bo.fun]))
    # save the whole bo object
    #with open("bo_efd.pkl", "wb") as f:
    #    pickle.dump(bo, f)