import pystorms
import numpy as np
import matplotlib.pyplot as plt


def controller(state):
    actions = np.zeros((2))
    for i in range(0, 2):
        if state[i] > 0.50:
            actions[i] = 0.50
        else:
            actions[i] = 0.00
    return actions
env = pystorms.scenarios.theta()
done = False

while not done:
    state = env.state()
    actions = controller(state)
    done = env.step(actions)

print(env.performance())
