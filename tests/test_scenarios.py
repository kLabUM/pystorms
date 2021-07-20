import pystorms
import numpy as np
import pytest


def test_alpha():
    """
    Tests for Alpha Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.alpha()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 18
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps

def test_beta():
    """
    Tests for Beta Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.beta()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 7
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps


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
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 11
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps

    # All valves closed - remanent water in basins
    env = pystorms.scenarios.gamma()
    done = False
    while not done:
        done = env.step(np.zeros(11))
    assert env.performance() > 10 ** 5


def test_theta():
    """
    Tests for Theta Scenario
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
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 18
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps


def test_epsilon():
    """
    Tests for Epsilon Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.epsilon()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 13
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps


def test_delta():
    """
    Tests for Delta Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.delta()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 6
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps

def test_zeta():
    """
    Tests for Zeta Scenario
    """
    # Initalize your environment
    env = pystorms.scenarios.zeta()
    done = False
    steps = 0
    while not done:
        state = env.state()
        # Check if performance measure is raising error
        if steps < 1:
            with pytest.raises(ValueError):
                env.performance()
        done = env.step()
        steps += 1
        # Check for length of state being returned
        assert len(state) == 6
        # Check if data log is working
        assert len(env.data_log["performance_measure"]) == steps
