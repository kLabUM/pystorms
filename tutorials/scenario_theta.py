import pystorms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def controller(
    depths: np.ndarray,
    tolerance: float = 0.50,
    LAMBDA: float = 0.50,
    MAX_DEPTH: float = 2.0,
) -> np.ndarray:
    """
    Implementation of equal-filling controller

    Parameters
    ----------
    depths: np.ndarray
        depths in the basins being controlled
    tolerance: float
        parameter to control oscillations in the actions
    LAMBDA: float
        parameter for tuning controller's response
    MAX_DEPTH: float
        max depth in the basins to compute filling degree

    Returns
    -------
    actions: np.ndarray
        control actions
    """

    # Compute the filling degree
    f = depths / MAX_DEPTH

    # Estimate the average filling degree
    f_mean = np.mean(f)

    # Compute psi
    N = len(depths)
    psi = np.zeros(N)
    for i in range(0, N):
        psi[i] = f[i] - f_mean
        if psi[i] < 0.0 - tolerance:
            psi[i] = 0.0
        elif psi[i] >= 0.0 - tolerance and psi[i] <= 0.0 + tolerance:
            psi[i] = f_mean

    # Assign valve positions
    actions = np.zeros(N)
    for i in range(0, N):
        if depths[i] > 0.0:
            k = 1.0 / np.sqrt(2 * 9.81 * depths[i])
            action = k * LAMBDA * psi[i] / np.sum(psi)
            actions[i] = min(1.0, action)
    return actions


def rule_based_controller(state):
    actions = np.zeros(2)
    actions = state/2.0 
    return actions


# Configure matplotlib style to make pretty plots
plt.style.use("seaborn-v0_8-whitegrid")

# Run simulation with gates open
env_uncontrolled = pystorms.scenarios.theta()

# Update the datalog to append states
env_uncontrolled.data_log["depthN"] = {}
env_uncontrolled.data_log["depthN"]['P1'] = []
env_uncontrolled.data_log["depthN"]['P2'] = []

done = False
while not done:
    done = env_uncontrolled.step()

# convert flows as a dataframe for easy plotting and timeseries handling
uncontrolled_flows = pd.DataFrame.from_dict(env_uncontrolled.data_log["flow"])
uncontrolled_flows.index = env_uncontrolled.data_log["simulation_time"]
uncontrolled_flows = uncontrolled_flows.resample("15min").mean()
uncontrolled_flows = uncontrolled_flows.rename(columns={"8": "Uncontrolled"})

uncontrolled_depth = pd.DataFrame.from_dict(env_uncontrolled.data_log["depthN"])
uncontrolled_depth.index = env_uncontrolled.data_log["simulation_time"]
uncontrolled_depth = uncontrolled_depth.resample("15min").mean()


# Controlled simulation
env_controlled = pystorms.scenarios.theta()
done = False

# Update the datalog to append states
env_controlled.data_log["depthN"] = {}
env_controlled.data_log["depthN"]['P1'] = []
env_controlled.data_log["depthN"]['P2'] = []

while not done:
    state = env_controlled.state()
    # Note the difference between controlled and uncontrolled simulation
    actions = controller(state)
    done = env_controlled.step(actions)

controlled_flows = pd.DataFrame.from_dict(env_controlled.data_log["flow"])
controlled_flows.index = env_controlled.data_log["simulation_time"]
controlled_flows = controlled_flows.resample("15min").mean()
controlled_flows = controlled_flows.rename(columns={"8": "Controlled"})

controlled_depth = pd.DataFrame.from_dict(env_controlled.data_log["depthN"])
controlled_depth.index = env_controlled.data_log["simulation_time"]
controlled_depth = controlled_depth.resample("15min").mean()

# Rule based controller
env_rule_controlled = pystorms.scenarios.theta()
done = False

# Update the datalog to append states
env_rule_controlled.data_log["depthN"] = {}
env_rule_controlled.data_log["depthN"]['P1'] = []
env_rule_controlled.data_log["depthN"]['P2'] = []

while not done:
    state = env_rule_controlled.state()
    actions = rule_based_controller(state)
    done = env_rule_controlled.step(actions)

env_rule_controlled_flows = pd.DataFrame.from_dict(env_rule_controlled.data_log["flow"])
env_rule_controlled_flows.index = env_rule_controlled.data_log["simulation_time"]
env_rule_controlled_flows = env_rule_controlled_flows.resample("15min").mean()
env_rule_controlled_flows = env_rule_controlled_flows.rename(columns={"8": "Rule-baseed Controller"})

env_rule_controlled_depth = pd.DataFrame.from_dict(env_rule_controlled.data_log["depthN"])
env_rule_controlled_depth.index = env_rule_controlled.data_log["simulation_time"]
env_rule_controlled_depth = env_rule_controlled_depth.resample("15min").mean()


fig, ax = plt.subplots(1, 3, sharey=True)
controlled_depth[['P1']].plot(ax=ax[0])
uncontrolled_depth[['P1']].plot(ax=ax[0])
env_rule_controlled_depth[['P1']].plot(ax=ax[0])

controlled_depth[['P2']].plot(ax=ax[1])
uncontrolled_depth[['P2']].plot(ax=ax[1])
env_rule_controlled_depth[['P2']].plot(ax=ax[1])

controlled_flows.plot(ax=ax[2])
uncontrolled_flows.plot(ax=ax[2])
env_rule_controlled_flows.plot(ax=ax[2])

fig.set_size_inches(8.0, 3.0)
plt.savefig("scenario_theta.svg", dpi=1000)
