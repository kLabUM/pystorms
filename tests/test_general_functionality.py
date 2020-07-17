import os
import pystorms
import numpy as np


def test_save():
    # pick a scenario
    env = pystorms.scenarios.theta()
    # run and create the data log
    done = False
    while not done:
        done = env.step()
    # save the data
    env.save(path="./temp.npy")
    # read the saved data
    data = np.load("./temp.npy", allow_pickle=True).item()
    assert len(data["performance_measure"]) == len(env.data_log["performance_measure"])
    os.remove("./temp.npy")


def test_save1():
    # pick a scenario
    env = pystorms.scenarios.theta()
    # run and create the data log
    done = False
    while not done:
        done = env.step()
    # save the data
    env.save()
    data = np.load("./data_theta.npy", allow_pickle=True).item()
    assert len(data["performance_measure"]) == len(env.data_log["performance_measure"])
    os.remove("./data_theta.npy")
