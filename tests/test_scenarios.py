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


def test_gamma_controller():
    """
    Test Gamma Controlled Performance
    1. Uncontrolled
    2. Closed
    """
    env = pystorms.scenarios.gamma()
    done = False
    while not done:
        done = env.step(np.ones(11))
    assert env.performance() > 10 ** 3
    # All valves closed - remanent water in basins
    env = pystorms.scenarios.gamma()
    done = False
    while not done:
        done = env.step(np.zeros(11))
    assert env.performance() > 10 ** 5


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
