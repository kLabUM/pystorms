"""Tests for the version and level keywords added in pystorms 2.0."""
import numpy as np
import pandas as pd
import pytest
import pyswmm.toolkitapi as tkai

import pystorms

V2_SCENARIOS = ["theta", "alpha", "gamma", "delta", "epsilon"]


def half_open(env):
    return np.ones(len(env.config["action_space"])) * 0.5


def run(env, controller, max_steps=None):
    """Drive *env* with *controller* until the event ends or max_steps is hit."""
    steps, done = 0, False
    while not done and (max_steps is None or steps < max_steps):
        done = env.step(controller(env.state()))
        steps += 1
    return steps, done


def first_fault(schedule):
    """Return (ID, stuck, fix) for the first entry of a fault schedule."""
    ID, windows = next(iter(schedule.items()))
    stuck, fix = windows[0]
    return ID, stuck, fix


def build_with_fault(level, attribute, seeds=range(40)):
    """Build a theta scenario with a seed whose *attribute* schedule is non empty."""
    for seed in seeds:
        np.random.seed(seed)
        env = pystorms.scenarios.theta(level=level)
        if getattr(env.env, attribute):
            return env
        env.terminate()
    pytest.fail("no seed produced a fault in {}".format(attribute))


# ----------------------------------------------------------------- versions
@pytest.mark.parametrize("name", V2_SCENARIOS)
def test_version_2_builds_and_runs(name):
    with getattr(pystorms.scenarios, name)(version="2") as env:
        n_states = len(env.config["states"])
        for _ in range(300):
            state = env.state()
            assert state.shape == (n_states,)
            assert np.all(np.isfinite(state))
            if env.step(half_open(env)):
                break
        assert len(env.data_log["performance_measure"]) > 0
        assert np.isfinite(env.performance())


def test_theta_version_2_is_harder():
    results = {}
    for version in ["1", "2"]:
        with pystorms.scenarios.theta(version=version) as env:
            steps, done = run(env, lambda state: np.ones(2) * 0.5)
            results[version] = (env.threshold, steps, env.performance())
    assert results["2"][0] == results["1"][0] / 2
    assert results["2"][1] < results["1"][1]  # the event ends sooner
    assert results["2"][2] > results["1"][2]  # the same control does worse


def test_version_2_reshapes_the_other_scenarios():
    with pystorms.scenarios.alpha(version="2") as env:
        assert env.config["action_space"] == [
            "Or1", "Or2", "Or3", "Or4", "Or5", "W1", "W2", "W3", "W4", "W5",
        ]
    with pystorms.scenarios.gamma(version="2") as env:
        assert [s[0] for s in env.config["states"]] == [
            "1", "2", "3", "4", "6", "7", "8", "10", "11",
        ]
        assert "O5" not in env.config["action_space"]
        assert "O9" not in env.config["action_space"]
        assert len(env.config["performance_targets"]) == 27
        assert env._performormance_threshold == 3.0
        assert len(env.state()) == 9
    with pystorms.scenarios.delta(version="2") as env:
        assert env.threshold == 0.5
        assert env.env.sim.end_time.day == 27
    with pystorms.scenarios.epsilon(version="2") as env:
        assert env._performormance_threshold == pytest.approx(1.075 * 0.7)
        assert env.env.sim.end_time.month == 2


def test_derived_network_and_run_files_go_to_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSTORMS_CACHE", str(tmp_path))
    with pystorms.scenarios.theta(version="2") as env:
        assert env.config["swmm_input"] == str(tmp_path / "theta_v2.inp")
        env.step(np.zeros(2))
    assert (tmp_path / "theta_v2.rpt").exists()

    with pystorms.scenarios.theta() as env:
        env.step(np.zeros(2))
    assert (tmp_path / "theta.rpt").exists()
    assert (tmp_path / "theta.out").exists()


# --------------------------------------------------------------- arguments
def test_integer_arguments_are_accepted():
    np.random.seed(0)
    with pystorms.scenarios.theta(version=2, level=2) as env:
        assert env.version == "2"
        assert env.level == "2"
        assert env.env.bias is not None
        # the level is remembered, it need not be passed again
        state = env.state()
        assert state.shape == (2,)
        assert not env.step(np.zeros(2))


@pytest.mark.parametrize(
    "kwargs", [dict(version="3"), dict(version=0), dict(level="4"), dict(level=0)]
)
def test_unsupported_version_or_level_raises(kwargs):
    with pytest.raises(ValueError):
        pystorms.scenarios.theta(**kwargs)
    # nothing was left open
    pystorms.scenarios.theta().terminate()


@pytest.mark.parametrize("name", ["beta", "zeta"])
def test_scenarios_without_a_second_version_refuse_it(name):
    with pytest.raises(ValueError):
        getattr(pystorms.scenarios, name)(version="2")


def test_level_cannot_be_raised_after_construction():
    with pystorms.scenarios.theta() as env:
        with pytest.raises(ValueError):
            env.state(level="2")
        with pytest.raises(ValueError):
            env.step(np.zeros(2), level="3")
        assert env.state(level="1").shape == (2,)


def test_true_state_is_available_at_higher_levels():
    np.random.seed(0)
    with pystorms.scenarios.theta(level="3") as env:
        for _ in range(300):
            env.step(np.ones(2) * 0.5)
        clean = env.state(level="1")
        assert np.array_equal(clean, env.state(level="1"))
        assert not np.array_equal(clean, env.state())


# ------------------------------------------------------------------ levels
def test_level_1_reports_the_true_state_and_draws_no_faults():
    with pystorms.scenarios.theta() as env:
        assert env.env.drift_rates is None
        assert env.env.bias is None
        assert env.env.actuator_schedule is None
        assert env.env.sensor_schedule is None
        assert np.array_equal(env.state(), env.state())


def test_faults_are_reproducible_with_a_seed():
    def draw():
        np.random.seed(11)
        with pystorms.scenarios.theta(level="3") as env:
            return (
                env.env.drift_rates.copy(),
                env.env.bias.copy(),
                env.env.actuator_schedule,
                env.env.sensor_schedule,
            )

    first, second = draw(), draw()
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[3] == second[3]


@pytest.mark.parametrize("level", ["2", "3"])
def test_degraded_readings_are_finite_and_differ_from_the_truth(level):
    np.random.seed(1)
    with pystorms.scenarios.theta(level=level) as env:
        noisy, clean = [], []
        done = False
        while not done:
            noisy.append(env.state())
            clean.append(env.state(level="1"))
            done = env.step(np.ones(2) * 0.5)
        noisy, clean = np.array(noisy), np.array(clean)
        assert np.all(np.isfinite(noisy))
        assert noisy.shape == clean.shape
        assert np.abs(noisy - clean).max() > 0
        assert np.isfinite(env.performance())


def test_stuck_actuator_holds_its_position():
    env = build_with_fault("2", "actuator_schedule")
    with env:
        asset, stuck, fix = first_fault(env.env.actuator_schedule)
        index = env.config["action_space"].index(asset)
        inside, followed = [], 0
        step, done = 0, False
        while not done:
            command = 0.3 if (step // 200) % 2 == 0 else 0.9
            done = env.step(np.ones(2) * command)
            step += 1
            if done:
                break
            now = env.env.getCurrentSimulationDateTime()
            position = env.env._getValvePosition(asset)
            if stuck < now < fix:
                inside.append(position)
            elif now > fix:
                followed += position == pytest.approx(command)
        assert len(inside) > 0
        assert len(set(inside)) == 1  # frozen for the whole window
        assert followed > 0  # and responsive again afterwards


def test_dropped_out_sensor_reports_zero():
    env = build_with_fault("3", "sensor_schedule")
    with env:
        sensor, stuck, fix = first_fault(env.env.sensor_schedule)
        index = [entry[0] for entry in env.config["states"]].index(sensor)
        inside, outside = [], []
        done = False
        while not done:
            now = env.env.getCurrentSimulationDateTime()
            reading = env.state()[index]
            (inside if stuck < now < fix else outside).append(reading)
            done = env.step(np.ones(2) * 0.5)
        assert len(inside) > 0
        assert all(reading == 0.0 for reading in inside)
        assert any(reading != 0.0 for reading in outside)


# ----------------------------------------------------------------- actions
def test_dict_actions_set_the_named_valves():
    with pystorms.scenarios.theta() as env:
        env.step({"1": 0.25, "2": 0.75})
        assert env.env._getValvePosition("1") == pytest.approx(0.25)
        assert env.env._getValvePosition("2") == pytest.approx(0.75)
        env.step(np.array([0.6, 0.1]))
        assert env.env._getValvePosition("1") == pytest.approx(0.6)
        assert env.env._getValvePosition("2") == pytest.approx(0.1)


def test_link_pollutant_reads_the_link():
    with pystorms.scenarios.epsilon() as env:
        for _ in range(500):
            env.step(np.ones(11) * 0.5)
        model = env.env.sim._model
        ids = model.getObjectIDList(tkai.ObjectType.POLLUT.value)
        expected = model.getLinkPollut("001", tkai.LinkPollut.linkQual)[ids.index("TSS")]
        assert env.env._getLinkPollutant("001", "TSS") == expected
        assert env.state()[-1] == expected


# --------------------------------------------------------------- lifetime
def test_terminate_is_idempotent_and_frees_the_engine():
    env = pystorms.scenarios.theta()
    env.step(np.zeros(2))
    env.terminate()
    env.terminate()
    with pystorms.scenarios.beta() as other:
        other.step(np.zeros(3))


def test_context_manager_closes_after_an_early_exit():
    with pytest.raises(RuntimeError):
        with pystorms.scenarios.theta() as env:
            env.step(np.zeros(2))
            raise RuntimeError("stop early")
    with pystorms.scenarios.theta() as env:
        assert env.state().shape == (2,)


def test_environment_reset_restarts_the_event():
    with pystorms.scenarios.theta() as env:
        for _ in range(20):
            env.step(np.zeros(2))
        state = env.env.reset()
        assert state.shape == (2,)
        assert not env.step(np.zeros(2))


# --------------------------------------------------------------- utilities
def test_append_rainfall(tmp_path):
    rain = pd.Series(
        [0.1, 0.2, 0.3], index=pd.date_range("2020-01-01", periods=3, freq="h")
    )
    path = pystorms.utilities.append_rainfall(
        pystorms.networks.load_network("theta"),
        rain,
        destination=str(tmp_path / "rain.inp"),
    )
    lines = open(path).read().splitlines()
    assert "[TIMESERIES] " in lines
    assert lines[-1].startswith("TestRain")
    assert lines[-1].rstrip().endswith("0.3")
