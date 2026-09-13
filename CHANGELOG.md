# Changelog

All notable changes to pystorms are recorded here. Earlier releases are
described on the [GitHub releases page](https://github.com/kLabUM/pystorms/releases).

## 2.0.0 - 2026-09-12

### Added

- `version` keyword on every scenario. `"1"` is the scenario as published;
  `"2"` is a harder variant of the same network for _theta_, _alpha_, _gamma_,
  _delta_ and _epsilon_. Derived networks are written to a per user cache
  directory (`PYSTORMS_CACHE`, `$XDG_CACHE_HOME/pystorms/networks` or
  `~/.cache/pystorms/networks`) rather than into the installed package.
- `level` keyword on every scenario. `"1"` reports the true state. `"2"` adds
  reading noise, sensor drift and calibration bias and lets actuators stick.
  `"3"` intensifies all of that and adds sensors that drop out. Faults are drawn
  from numpy's global random state when the scenario is built, so seeding
  before construction reproduces them.
- `scenario.terminate()` and context manager support (`with
  pystorms.scenarios.theta() as env:`), so an episode that stops early can be
  closed and the next scenario built in the same process.
- `pystorms.networks.cache_dir()` and `run_file_paths()` helpers.
- The controller implementations, tuned parameters and analysis scripts behind
  the accompanying manuscript, under `baseline_controllers/`.
- `tutorials/Versions_and_Levels.ipynb`.
- Tests for versions, levels, dict actions, scenario lifetime and utilities.
- A changelog.

### Changed

- The level passed when building a scenario is remembered. `state()` and
  `step()` apply it by default; passing `level` to them is now optional and
  only needed to read the true state (`level="1"`) from a degraded scenario.
  Asking for a higher level than the scenario was built with raises
  `ValueError` instead of `AttributeError`.
- `version` and `level` are validated. Integers are accepted and normalised to
  strings; unsupported values raise `ValueError` instead of silently running
  version 1, crashing later, or (for _beta_ and _zeta_) pretending a second
  version exists.
- SWMM report (`.rpt`) and binary output (`.out`) files are written to the
  cache directory instead of next to the shipped input files inside the
  installed package.
- At level 3 a dropped out sensor now reports exactly zero. Previously the
  zero was applied before noise and drift, so the reading was noise around
  zero rather than the flat zero described in the documentation.
- Level 1 no longer draws from the random number generator on every
  `state()` call.
- `pyswmm` 2.x is supported. pystorms keeps pyswmm's single simulation
  bookkeeping in step so scenarios can be built one after another, and its
  context manager warning is silenced since pystorms manages the simulation
  lifetime itself.
- Packaging metadata moved from `setup.py` to `pyproject.toml`, with an SPDX
  license expression, project URLs and Python 3.9 to 3.14 declared.
- Continuous integration runs on uv managed interpreters across Linux, macOS
  and Windows, and releases are published to PyPI through trusted publishing
  on tag pushes.
- Fixed out of range infiltration parameters in the _delta_ network.
- The project website moved from GitHub hosting at pystorms.org to Netlify, at
  https://pystorms.netlify.app. Links in the README and the package metadata
  point there.

### Fixed

- Passing actions as a dict set the valves to their position in the dict
  (0, 1, 2, ...) instead of the given values.
- _epsilon_ read the TSS concentration of node 001 instead of conduit 001, so
  the pollutant state and the loading objective now use the link value. The
  difference is small (about a tenth of a percent at the outlet) but changes
  the reported performance numbers slightly.
- `utilities.append_rainfall` failed under pandas 3 (positional indexing on a
  Series with a datetime index).
- `utilities._to_dataframe` referenced an undefined name and pandas was not
  imported.
- `environment(..., ctrl=False)` passed a keyword pyswmm 2 no longer accepts.
- Configuration files were opened without being closed.
