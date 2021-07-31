import pystorms
import numpy as np
import pytest


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
