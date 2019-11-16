import pystorms
import numpy as np
import pytest


def test_gamma():
    """
    Tests for Gamma Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.gamma()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        actions = np.ones(11)
        done = env.step(actions)
        steps += 1
        # Check for length of state being returned
        assert len(state) == 11
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps


def test_theta():
    """
    Tests for Alpha Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.theta()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        actions = np.ones(2)
        done = env.step(actions)
        steps += 1
        # Check for length of state being returned
        assert len(state) == 2
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps
