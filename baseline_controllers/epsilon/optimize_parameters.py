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

# EPSILON
evaluating = "both" # "constant-flow" or "efd" or "both"
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
    for state in env.config['states']:
        if 'depth' in state[1]:
            node_id = state[0]
            max_depths[node_id] = pyswmm.Nodes(env.env.sim)[node_id].full_depth
            max_depths_array = np.append(max_depths_array, pyswmm.Nodes(env.env.sim)[node_id].full_depth)
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
                for i in range(len(u_open_pct)):
                    if u_open_pct[i,0]< 0.09:
                        u_open_pct[i,0] = 0.09
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
                
        else:
            done = env.step(u_open_pct.flatten())
            

    return {"cost": sum(env.data_log["performance_measure"]), "final_depths": final_depths}


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

if evaluating == "constant-flow":
    domain = []
    x0 = []
    for i in range(1, 12):
        domain.append(Real(0.3,5.0,name=str(i)))
        x0.append(2.0)
    
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=3, n_initial_points=2, 
                     initial_point_generator = 'lhs',verbose=True)
    #bo = gp_minimize(f_constant_flows, domain,x0=x0, verbose=True)
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal constant flows
    np.savetxt(str("v" + version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" + version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))

    # save the whole bo object
    #with open("bo_constant_flows.pkl", "wb") as f:
    #    pickle.dump(bo, f)


elif evaluating == "efd":
    optimal_constant_flows = np.loadtxt(str("v" + version +"/optimal_constant_flows.txt"))
    #domain = [{"name": "TSS_feedback", "type": "continuous", "domain": (-1, -1e-4)},
    #        {"name": "efd_gain", "type": "continuous", "domain": (0.0, 1.0)}]
    domain = [Real(-1e-1,-1e-4, name="TSS_feedback"), Real(0.0, 1.0, name="efd_gain")]
    x0 = [-3e-3, 0.25] # manual optimal was -3e-3, and 0.25 with the old constant heads
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=3, n_initial_points=2, 
                     initial_point_generator = 'lhs',verbose=True)
    
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
    
elif evaluating == 'both':
    evaluating = "constant-flow"
    domain = []
    x0 = []
    for i in range(1, 12):
        domain.append(Real(0.5,6.0,name=str(i)))
        x0.append(4.5)
    
    bo = gp_minimize(f_constant_flows, domain,x0=x0, 
                     n_calls=1000, n_initial_points=100, 
                     initial_point_generator = 'lhs',verbose=True)
    #bo = gp_minimize(f_constant_flows, domain,x0=x0, verbose=True)
    print(bo.x)
    print(bo.fun)
    print(bo.x_iters)
    # save the optimal constant heads 
    np.savetxt(str("v" + version +"/optimal_constant_flows.txt"), bo.x)
    # save the float bo.fun to a text file 
    np.savetxt(str("v" + version +"/optimal_constant_flows_cost.txt"), np.array([bo.fun]))

    # save the whole bo object
    #with open("bo_constant_flows.pkl", "wb") as f:
    #    pickle.dump(bo, f)


    evaluating = "efd"
    optimal_constant_flows = np.loadtxt(str("v" + version +"/optimal_constant_flows.txt"))
    domain = [Real(-1,-1e-5, name="TSS_feedback"), Real(0.0, 1.0, name="efd_gain")]
    x0 = [-1e-5, 1e-2] # manual optimal was 3e-3, and 0.25
    
    bo = gp_minimize(f_efd, domain, x0=x0, n_calls=500, n_initial_points=50, 
                     initial_point_generator = 'lhs',verbose=True)
    
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
    