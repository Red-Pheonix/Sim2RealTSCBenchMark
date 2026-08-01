# Sim2Real Traffic Signal Control Benchmark

A benchmark for the sim-to-real gap in reinforcement-learning traffic signal
control. Policies are trained in CityFlow (sim) and evaluated in SUMO (real)
under controlled perturbations of each element of the MDP — **observations**,
**transitions**, **actions** (execution delay and phase-transition structure),
and **rewards** — together with mitigation methods for each gap.

# Installation

Linux only (tested on Ubuntu; CityFlow does not build on Windows).
Python >= 3.10.

## One-click

Activate a fresh environment, then:

```
python3 -m venv .venv && source .venv/bin/activate
bash install.sh
```

This installs the python dependencies, the SUMO python bindings (the `libsumo`
wheel bundles the simulator — no system SUMO install needed), builds CityFlow
from source (requires `build-essential` and `cmake`), and smoke-tests the
imports.

## Manual

### CityFlow

CityFlow 0.1 is used for the experiments (see the
[CityFlow docs](https://cityflow.readthedocs.io/en/latest/install.html#)):

```
sudo apt update && sudo apt install -y build-essential cmake
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .
```

Test: `python -c "import cityflow; cityflow.Engine"`

### SUMO

SUMO 1.26.0 is used for the experiments, through the `libsumo` python bindings:

```
pip install libsumo==1.26.0 traci==1.26.0
```

(A system-wide SUMO — `sudo add-apt-repository ppa:sumo/stable && sudo apt-get
install sumo sumo-tools` — is optional; the experiments run entirely through
libsumo.)

Test: `python -c "import libsumo, traci"`

### Python dependencies

```
pip install -r requirements.txt
```

# Running a single experiment

Every experiment is one command: pick a gap (runner), an agent, a network, a
mitigation method, and a gap setting.

```
python run_s2r_actions.py -a dqn -n tempe_1x1 --act_model direct_transfer \
    --real_setting setting2 --prefix my_run
```

`direct_transfer` evaluates the committed pretrained policy (`pretrained/tsc`)
on the real side with zero adaptation — no training, finishes in seconds. Any
other method trains first (minutes to hours depending on the network).

| gap | runner | method flag | methods |
|---|---|---|---|
| observations | `run_s2r_observations.py` (add `--real_world sumo`) | `--obs_model` | `direct_transfer`, `domain_randomization`, `vae`, `darla`, `atc`, `lusr`, `recon_baseline` |
| transitions | `run_s2r.py` | `-gt` | `direct_transfer`, `domain_randomization`, `domain_adaptation`, `gat`, `ugat`, `jlgat` |
| actions | `run_s2r_actions.py` | `--act_model` | `direct_transfer`, `naive`, `delayed_q`, `oblivious_q`, `prlight`, `dr`, `dr_noshield`, `gat`, `gat_shield`, `ugat`, `ugat_shield` |
| rewards | `run_s2r_rewards.py` | `--reward_model` | `direct_transfer`, `reward_inference`, `morl_grid`, `dynamic_reward_shaping`, `reward_oracle` |

- **Agents** (`-a`): `dqn`, `presslight` (RL); `fixedtime`, `maxpressure` (non-RL baselines, `direct_transfer` only).
- **Networks** (`-n`): `tempe_1x1`, `bullhead_1`, `cologne1`, `ingolstadt1`, `hz1x1` (single-intersection); `tempe_16`, `bullhead_3`, `cologne3`, `ingolstadt7`, `hz4x4` (multi).
- **Settings** (`--real_setting`): a YAML under `configs/<task>/settings/` defining the gap itself —
  observation corruption (`noise3…noise20`, `dz10…dz100`, `sensor5…sensor70`, `combine1…4`),
  transition dynamics (`setting1…4` = light/heavy load, rain, snow),
  actuation delay (`setting1…4` = 20/30/40/60 s) or phase-transition structure
  (`cyclic`, `flexible`, `barrier_leading_fixed`, `barrier_lagging_fixed`, `barrier_leading_lagging_fixed`),
  and hidden real reward (`efficiency_aligned`, `emission_heavy`, `fairness_heavy`, `physical_safety_heavy`).

Results land in
`data/output_data/<task>/cityflow_<agent>/<network>/<prefix>/logger/` as a
tab-separated `*_DTL.log` (one row per evaluation; the `REAL_TEST` rows are the
real-side numbers) plus a `*_BRF.log` with per-episode detail. The command
above prints a row that matches the shipped log for the same cell under
`logs/sim2real_actions/`.

Two batch scripts reproduce whole reference blocks from the committed weights
alone: `scripts/run_baseline_evals.sh` (the pretrained policies in both
engines — the sim/real reference lines) and `scripts/run_nonrl_gap_evals.sh`
(fixedtime/maxpressure across all 33 gap settings).

# Tables and figures

`make_figures.ipynb` is the one entry point from raw logs to paper numbers: its
first cell rebuilds every `tables/*.csv` from the run logs shipped in `logs/`
(via `scripts/gap_tables.py`, the single source of truth for the selection
rules), and the remaining cells build the paper figures into `Figures/`. A
fresh clone runs it top to bottom with no other inputs.

The four `analyze_*.ipynb` notebooks are per-gap exploratory companions
(availability matrices, per-network pivots, per-checkpoint travel-time traces);
they share the same `scripts/gap_tables.py` builders, so their numbers are the
paper numbers by construction.
