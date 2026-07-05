"""Base trainer for the reward gap (naive / direct transfer).

Sim = cityflow, real = sumo (the intended sim2real setup). The agent trains in sim
on a TRAINING reward and is evaluated zero-shot in real, where it is scored by the
hidden true objective `R_real` (a linear cost over the feature bank, weights `w*` in
the setting file -- the agent never sees them). This base class IS the **naive**
baseline: it trains on the agent's native proxy reward (pressure / lane-waiting).

Method trainers subclass this and override one of:
  * `build_reward_transform(feature_bank)` -> a callable `Φ -> per-agent reward`
    (reward_shaping / reward_random), or
  * `train()` for methods that spend the real budget (reward_inference /
    dynamic_reward_shaping / morl_grid).

Budget: one fungible pool of 300 episodes per method (sim + real <= 300), of which at
most `real_episodes` (100) may be REAL. `sim_episodes` (300) is the naive-family training
length and the pool reference; a method is defined by how it converts real budget into
information. Every real rollout that shapes the deployed policy (probes, grid/BO selection
evals, keep-best validations) is charged to the real budget via `self._real_rollouts`;
scoring-only evals (final test, the naive-family `train_test` transfer curve) pass
`count_budget=False` and are free. naive spends 0 real -- it gets the whole pool in sim.
See notes/reward_gap_fix_plan.md (Task 7).
"""

import os

import numpy as np
from tqdm import tqdm

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer
from trainer.rewards.feature_bank import CORE_COMPONENTS, FeatureBank
from trainer.rewards.reward_transforms import DEFAULT_COMPONENT_NORM, TrueReward


# --- Data layer: which world info function backs each non-core ("extra") cost ---
# Core components (queue/delay/waiting/pressure/switches) are lane/phase based and
# computable on every world. The extras below need a world info function; an extra is
# available on a world iff that function is registered there -- cityflow registers only
# `intersection_fairness`, sumo registers all of them. `safety` is special: it has no
# info function (the PT/shield trainer injects it), so it's never resolved here.
EXTRA_INFO_FN = {
    "fairness": "intersection_fairness",   # cross-sim (both worlds)
    "emission": "lane_co2",                # sumo-only
    "fuel": "lane_fuel",                   # sumo-only
    "emergency_stops": "intersection_emergency_stops",  # sumo-only
    "collisions": "intersection_collisions",            # sumo-only
    # surrogate-safety conflicts (TTC < SSM_TTC_THRESHOLD, 1.5s SSAM standard); registered on
    # the sumo world only when the device is enabled (rewards create_world passes
    # ssm_device=True), so availability resolution handles it like the other extras
    "ssm_conflicts": "intersection_ssm_conflicts",       # sumo-only, device-gated
}

# DTL `mode` values (column 2) -- the repo-wide canonical four:
#   SIM_TRAIN  -- sim training episode (no real interaction)
#   SIM_TEST   -- sim evaluation rollout (unused by the reward gap today)
#   REAL_TRAIN -- real rollout whose outcome FEEDS the method (probe / grid or BO
#                 selection eval / keep-best validation). This is the real-budget
#                 spend: count(REAL_TRAIN) == self._real_rollouts <= real_episodes.
#   REAL_TEST  -- real scoring eval (naive-family transfer curve + final test); free.
#
# DTL data-log column order (header row written once at the top of each run's log).
# First 9 match the other tasks' DTL; then the reward-gap extras. The raw-metric
# columns are appended AFTER `detail` so positional parsers of the old 13-col schema
# (e.g. scripts/rescore_norms.py, cols[11]/cols[12]) keep working unchanged.
#
# Raw benchmark-metric columns + units (per rollout the row reports; an EMPTY cell
# means this side's simulator cannot compute the metric -- distinct from a true 0.
# cityflow rows carry only `fairness`; sumo rows carry all five). Units verified
# against the sumo 1.26 TraCI docs ("Sum of CO2 emissions on this lane in mg/s
# during this time step"; likewise fuel since sumo 1.14) and env.step == 1
# simulated second (all shipped sumocfgs use step-length 1):
#   fairness         veh    final max-min spread of CUMULATIVE served throughput
#                           (vehicle counts) across demand-active approaches, mean
#                           over intersections (Raeis & Leon-Garcia arXiv:2107.10146).
#                           Grows with episode length -- only compare equal-length runs.
#   emission         kg     episode-total CO2, sum over vehicles on the CONTROLLED
#                           in+out lanes (junction-internal lanes excluded), sumo
#                           HBEFA model; mg/s summed per 1s step.
#   fuel             kg     same accumulation for fuel (mg/s since sumo 1.14);
#                           ~/0.74 kg/L for liters of gasoline. Sanity: fuel/emission
#                           ~= 0.32 (stoichiometric CO2 of gasoline).
#   emergency_stops  count  episode-total sumo "emergency stop" events: a vehicle
#                           forced to an abrupt stop exceeding its emergencyDecel
#                           (e.g. caught at the stop bar by a phase switch); one
#                           event per occurrence, attributed to the controlling
#                           intersection.
#   ssm_conflicts    count  episode-total vehicles whose SSM-device min TTC dropped
#                           below SSM_TTC_THRESHOLD (world_sumo.py; 1.5 s = the FHWA
#                           SSAM standard since 2026-07-05, earlier runs used a 3.0 s
#                           screening value -- counts NOT comparable across the two).
#                           Each vehicle counted once per episode. The STANDARD
#                           surrogate-safety measure; sumo traffic is collision-free
#                           by construction so near-crash conflicts, not crashes,
#                           carry the safety signal.
#   collisions       count  episode-total UNIQUE collider-victim pairs from
#                           simulation.getCollisions. Junction-conflict detection is
#                           enabled by world_sumo's sumo_cmd (--collision.check-
#                           junctions, action=warn: detection only, physics
#                           untouched); pairs dedup per episode in
#                           get_intersection_collisions. Runs BEFORE 2026-07-04 had
#                           detection off (sumo default) -> structurally 0.
#   phi_raw          --     RAW unnormalized per-decision feature-bank costs as
#                           `name=value;...` (summed over intersections, mean over
#                           decisions) -- the numbers BEHIND the `components` column,
#                           which stays the WEIGHTED normalized w*_i*(phi_i/n_i)
#                           terms. Eval rows only (empty on SIM_TRAIN). Per-term
#                           units (per intersection-decision): queue = veh halted,
#                           mean over incoming lanes; delay = 1 - mean_speed/limit
#                           in [0,1]; waiting = veh*s (SUM of currently-waiting
#                           vehicles' accumulated waits), mean over incoming lanes;
#                           pressure = |veh in-out|; switches / safety = 0/1 flag
#                           (mean over decisions = rate in [0,1]). The info-backed
#                           terms are INTERVAL-ACCURATE (FeatureBank.step_accumulate,
#                           every sim step): emission / fuel / fairness = interval
#                           MEAN (g/s resp. veh; mean over incoming lanes for the
#                           lane-keyed ones); emergency_stops / collisions = interval
#                           SUM (events this decision) -- so summed over an episode's
#                           decisions the event terms EQUAL the episode metric
#                           columns (up to lane attribution), and R_real scores full
#                           coverage rather than a 1-in-10 boundary point-sample.
RAW_METRIC_COLUMNS = ["fairness", "emission", "fuel", "emergency_stops",
                      "ssm_conflicts", "collisions"]
DTL_COLUMNS = [
    "exp_name", "mode", "step", "travel_time", "loss", "rewards",
    "queue", "delay", "throughput", "train_reward", "R_real", "components", "detail",
    *RAW_METRIC_COLUMNS, "phi_raw",
]


# --- Raw benchmark metrics logged alongside R_real (logging-only) -------------
# metric name -> world info fn. Deliberately DECOUPLED from the feature bank: the
# bank only tracks the setting's components (touching it would change method
# behavior -- reward_inference probes per component, random_reward's simplex, ...),
# while these are recorded on EVERY run regardless of what R_real weights.
RAW_METRIC_FNS = {
    "fairness": "intersection_fairness",                 # cross-sim
    "emission": "lane_co2",                              # sumo-only
    "fuel": "lane_fuel",                                 # sumo-only
    "emergency_stops": "intersection_emergency_stops",   # sumo-only
    "collisions": "intersection_collisions",             # sumo-only
    "ssm_conflicts": "intersection_ssm_conflicts",       # sumo-only, device-gated
}


class RawMetricsAccumulator:
    """Episode accumulator for the raw benchmark metrics (see RAW_METRIC_FNS).

    One per side (sim / real). Subscribes every metric fn the wrapped world
    registers (the world computes subscribed fns once per step via its
    subscribe/_update_infos path, so `step()` only reads the cached value) and
    folds it into episode totals:

      * emission / fuel -- lane-keyed rates (g/s over controlled lanes); summed
        over lanes and steps (1 s world interval) -> episode totals, reported in kg.
      * emergency_stops / collisions -- per-intersection counts this step ->
        episode-total counts.
      * fairness -- the world accumulates served throughput internally, so the fn
        returns a CUMULATIVE max-min spread; keep the LAST step's mean over
        intersections (veh).

    Call `reset()` after `env.reset()`, `step()` after every `env.step()`, and
    `values()` at rollout end. Metrics the world can't compute are simply absent
    from `values()` (logged as an empty DTL cell, never a fake 0).
    """

    def __init__(self, world):
        self.world = world
        self.fns = {
            m: fn for m, fn in RAW_METRIC_FNS.items()
            if fn in getattr(world, "info_functions", {})
        }
        if self.fns:
            world.subscribe(list(self.fns.values()))
        self.reset()

    def reset(self):
        self._totals = {m: 0.0 for m in self.fns}

    def step(self):
        for m, fn in self.fns.items():
            vals = self.world.get_info(fn)
            total = float(sum(vals.values())) if vals else 0.0
            if m == "fairness":
                # cumulative world-side state: keep the latest per-intersection mean
                self._totals[m] = total / max(len(vals), 1) if vals else 0.0
            else:
                self._totals[m] += total

    def values(self):
        out = dict(self._totals)
        for m in ("emission", "fuel"):
            if m in out:
                out[m] /= 1000.0  # accumulated grams -> kg per episode
        return out


def resolve_components(world, extra_components):
    """Resolve, for a given world, which components it can actually compute and the
    world info fn backing each available extra. Returns (available_list, info_fns).
    This is the single place that knows simulator capabilities -- the FeatureBank
    just receives the result."""
    available = list(CORE_COMPONENTS)
    info_fns = {}
    world_fns = getattr(world, "info_functions", {})
    for c in extra_components or []:
        fn = EXTRA_INFO_FN.get(c)
        if fn is not None and fn in world_fns:
            available.append(c)
            info_fns[c] = fn
    return available, info_fns


@Registry.register_trainer("sim2real_rewards")
class Sim2RealRewardsTrainer(BaseTrainer):
    """Reward-gap trainer; cityflow sim rollouts, sumo real rollouts."""

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "naive"

        cmd_args = Registry.mapping["command_mapping"]["setting"].param
        trainer_args = Registry.mapping["trainer_mapping"]["setting"].param
        logger_args = Registry.mapping["logger_mapping"]["setting"].param

        self.cityflow_path = os.path.join(
            "configs/sim", "cityflow", cmd_args["network"] + ".cfg"
        )
        self.sumo_path = os.path.join(
            "configs/sim", "sumo", cmd_args["network"] + ".cfg"
        )
        self.steps = trainer_args["steps"]
        self.test_steps = trainer_args["test_steps"]
        self.buffer_size = trainer_args["buffer_size"]
        self.action_interval = trainer_args["action_interval"]
        self.save_rate = logger_args["save_rate"]
        self.learning_start = trainer_args["learning_start"]
        self.update_model_rate = trainer_args["update_model_rate"]
        self.update_target_rate = trainer_args["update_target_rate"]
        self.test_when_train = trainer_args["test_when_train"]
        # Thin the naive-family transfer curve (REAL_TEST rows) like the other gaps:
        # eval every `real_eval_interval` episodes (300 eps / 3 -> 100 rows). 0 = every
        # episode (legacy behavior).
        self.real_eval_interval = trainer_args.get("real_eval_interval", 0)
        self.yellow_time = trainer_args["yellow_length"]

        sim2real_config = self.get_sim2real_config()
        self.load_pretrained = sim2real_config.get("load_pretrained", False)
        # Budget: real <= sim. sim_episodes drives sim training; real_episodes is the
        # real-rollout budget a method may spend (naive ignores it).
        self.sim_episodes = int(
            sim2real_config.get("sim_episodes", trainer_args["episodes"])
        )
        # Direct transfer: pretrained policy, ZERO training, one real eval. The flag
        # lives in the method yml (settings override sim2real keys, so a plain
        # `sim_episodes: 0` there would be clobbered by the setting's 300).
        if sim2real_config.get("direct_transfer", False):
            self.sim_episodes = 0
        # real_episodes is the REAL-rollout CAP (<=100 under the budget policy), a hard
        # ceiling a method enforces against -- NOT bound by sim_episodes. (The old
        # `real <= sim` clamp belonged to the retired split-budget framing; under the
        # 300-pool policy real can exceed sim while still fitting the pool.)
        self.real_episodes = int(sim2real_config.get("real_episodes", 0))
        self.episodes = self.sim_episodes  # back-compat for any base hooks

        # Hidden true objective + which extended components are in play.
        self.true_reward_weights = sim2real_config.get("true_reward", {})
        self.extra_components = sim2real_config.get("reward_components", [])
        # Per-component normalizers (config-driven; base.yml holds the values, settings
        # may override). Merge over the code defaults so a partial config still works.
        self.component_norm = {
            **DEFAULT_COMPONENT_NORM,
            **(sim2real_config.get("component_norm") or {}),
        }

        # exp_name has FOUR components: network _ setting _ agent _ mitigation-method
        # (the reward_model). self.method isn't set yet here -- subclasses assign it
        # after super().__init__() -- so read the method from the command args.
        reward_model = cmd_args.get("reward_model") or "naive"
        self.exp_name = (
            f'{cmd_args["network"]}_{cmd_args["real_setting"]}'
            f'_{cmd_args["agent"]}_{reward_model}'
        )
        self.model_dir = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            Registry.mapping["logger_mapping"]["setting"].param["model_dir"],
            self.exp_name,
        )

        base_log_name = os.path.basename(
            self.logger.handlers[-1].baseFilename
        ).removesuffix("_BRF.log")
        self.log_file = os.path.join(
            Registry.mapping["logger_mapping"]["path"].path,
            logger_args["log_dir"],
            base_log_name + "_DTL.log",
        )
        # DTL header (written once at the top of the fresh per-run data log).
        with open(self.log_file, "w") as f:
            f.write("\t".join(DTL_COLUMNS) + "\n")
        # Monotonic sim-episode counter so every method's SIM_TRAIN rows share one
        # step axis (warmup + per-candidate fine-tunes all advance it).
        self._sim_train_step = 0
        self._last_train_reward = 0.0
        # Real-budget counter: every real rollout whose outcome shapes the deployed
        # policy (probes, grid/BO selection evals, keep-best validations) counts against
        # `real_episodes` (the <=100 cap). Scoring-only evals (final test, naive-family
        # `train_test` transfer curve) pass count_budget=False and don't. See
        # notes/reward_gap_fix_plan.md Task 7.
        self._real_rollouts = 0

        self.world_sim = None
        self.world_real = None
        self.agents_sim = None
        self.agents_real = None
        self.metric_sim = None
        self.metric_real = None
        self.env_sim = None
        self.env_real = None
        self.total_decision_num_sim = 0

        self.create()

        # Feature banks (one per side) + the eval scorer over the REAL bank. The trainer
        # resolves availability per world (data layer) and hands the bank the result, so
        # the bank stays simulator-agnostic.
        avail_sim, fns_sim = resolve_components(self.world_sim, self.extra_components)
        avail_real, fns_real = resolve_components(self.world_real, self.extra_components)
        self.feature_bank_sim = FeatureBank(
            self.world_sim, self.agents_sim, self.extra_components,
            available=avail_sim, info_fns=fns_sim,
        )
        self.feature_bank_real = FeatureBank(
            self.world_real, self.agents_real, self.extra_components,
            available=avail_real, info_fns=fns_real,
        )
        self.components = self.feature_bank_real.components
        w_star = self.feature_bank_real.weight_vector(self.true_reward_weights)
        self.true_reward = TrueReward(w_star, self.components, norm=self.component_norm)

        # Raw-metrics recorders (logging only; see RawMetricsAccumulator). One per
        # side; the rollout loops pick by env identity and stash the last rollout's
        # values here for writeLog. `_last_phi_raw` is the raw unnormalized
        # per-decision phi of the last EVAL rollout (phi_raw column).
        self._raw_acc_sim = RawMetricsAccumulator(self.world_sim)
        self._raw_acc_real = RawMetricsAccumulator(self.world_real)
        self._last_raw_metrics = {}
        self._last_phi_raw = {}

        # Training reward transform: None -> naive (native proxy reward). Method
        # subclasses return a callable `Φ -> per-agent reward`.
        self.reward_transform = self.build_reward_transform(self.feature_bank_sim)

        self.world = self.world_real
        self.agents = self.agents_real
        self.metric = self.metric_real
        self.env = self.env_real

    # ---- mitigation hooks -------------------------------------------------
    def build_reward_transform(self, feature_bank):
        """Return the TRAINING reward transform (callable `Φ -> per-agent reward`),
        or None for naive (train on the agent's native proxy reward)."""
        return None

    def on_episode_start(self, episode):
        """Per-episode hook (reward_random resamples its weights here)."""

    # ---- config -----------------------------------------------------------
    def get_sim2real_config(self):
        sim2real_setting = Registry.mapping.get("sim2real_mapping", {}).get("setting")
        if sim2real_setting and hasattr(sim2real_setting, "param"):
            return sim2real_setting.param
        return {}

    def _build_world_kwargs(self):
        return {
            "interface": Registry.mapping["command_mapping"]["setting"].param[
                "interface"
            ]
        }

    def create_world(self):
        thread_num = Registry.mapping["command_mapping"]["setting"].param["thread_num"]
        interface = Registry.mapping["command_mapping"]["setting"].param["interface"]
        self.world_sim = Registry.mapping["world_mapping"]["cityflow"](
            self.cityflow_path, thread_num
        )
        # ssm_device: equip vehicles with the SSM device so the real side measures
        # surrogate-safety conflicts (rewards task only -- costs sim time).
        self.world_real = Registry.mapping["world_mapping"]["sumo"](
            self.sumo_path, **{"interface": interface, "ssm_device": True}
        )

    def create_agent_world(self, world):
        agents = []
        agent = Registry.mapping["model_mapping"][
            Registry.mapping["command_mapping"]["setting"].param["agent"]
        ](world, 0)
        num_agent = int(len(world.intersections) / agent.sub_agents)
        agents.append(agent)
        for i in range(1, num_agent):
            agents.append(
                Registry.mapping["model_mapping"][
                    Registry.mapping["command_mapping"]["setting"].param["agent"]
                ](world, i)
            )
        if Registry.mapping["model_mapping"]["setting"].param["name"] == "magd":
            for ag in agents:
                ag.link_agents(agents)
        return agents

    def pretrained_model_dir(self):
        return os.path.join(
            "pretrained",
            "tsc",
            Registry.mapping["command_mapping"]["setting"].param["agent"],
            Registry.mapping["command_mapping"]["setting"].param["network"],
        )

    def create_agents(self):
        self.agents_sim = self.create_agent_world(self.world_sim)
        self.agents_real = self.create_agent_world(self.world_real)
        if Registry.mapping["model_mapping"]["setting"].param["load_model"]:
            self.load_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)

    def create_metrics(self):
        if Registry.mapping["command_mapping"]["setting"].param["delay_type"] == "apx":
            lane_metrics = ["rewards", "queue", "delay"]
            world_metrics = ["real avg travel time", "throughput"]
        else:
            lane_metrics = ["rewards", "queue"]
            world_metrics = ["delay", "real avg travel time", "throughput"]
        self.metric_sim = Metrics(
            lane_metrics, world_metrics, self.world_sim, self.agents_sim
        )
        self.metric_real = Metrics(
            lane_metrics, world_metrics, self.world_real, self.agents_real
        )

    def create_env(self):
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        self.env_real = TSCEnv(self.world_real, self.agents_real, self.metric_real)

    def load_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.load_model(model_dir, e)

    def save_agents(self, agents, model_dir, e=None):
        for ag in agents:
            ag.save_model(model_dir, e)

    def set_replay(self, env, suffix, enabled):
        return

    def _capture_reset_base(self):
        """Snapshot the current sim policy so subsequent `_reset_sim_policy` calls REUSE
        these weights (warm-start) instead of a fresh random init. Call once after the
        pretrained model is loaded, so probe / grid policies start from the SAME place as
        naive -- otherwise they retrain from scratch and are unfairly handicapped."""
        base_dir = self.model_dir + "_reset_base"
        self.save_agents(self.agents_sim, base_dir)
        self._reset_base_dir = base_dir

    def _reset_sim_policy(self, warm_start=True):
        """Re-create the sim agents and rewire env/metric/feature bank to them, for an
        INDEPENDENT policy (reward_inference probes/final, morl_grid per-grid). With
        `warm_start` (default) the new agents REUSE the captured base weights when
        available (warm-start from pretrained; see `_capture_reset_base`), so a SCORED
        policy starts where naive does rather than from random init. Pass
        `warm_start=False` for reward_inference PROBES: their job is to induce DIVERSE
        feature profiles for identification, which a shared warm init destroys (every
        probe converges to the same policy) -- they want independent random inits."""
        self.agents_sim = self.create_agent_world(self.world_sim)
        if warm_start and getattr(self, "_reset_base_dir", None) is not None:
            self.load_agents(self.agents_sim, self._reset_base_dir)
        self.metric_sim = Metrics(
            self.metric_sim.lane_metric_List,
            self.metric_sim.world_metrics,
            self.world_sim,
            self.agents_sim,
        )
        self.env_sim = TSCEnv(self.world_sim, self.agents_sim, self.metric_sim)
        # Rebuild the bank with the SAME resolved availability/info-fns as the initial
        # build (otherwise a reset silently drops fairness + the sumo info-backed costs).
        avail_sim, fns_sim = resolve_components(self.world_sim, self.extra_components)
        self.feature_bank_sim = FeatureBank(
            self.world_sim, self.agents_sim, self.extra_components,
            available=avail_sim, info_fns=fns_sim,
        )
        self.total_decision_num_sim = 0

    # ---- feature helpers --------------------------------------------------
    def _stash_raw(self, raw_acc, feature_bank, phi_sum, decisions):
        """Stash an eval rollout's raw metrics + raw per-decision phi for the DTL row
        that follows (writeLog reads these). Shared by every eval loop (base / PT
        override / reward_inference probes)."""
        self._last_raw_metrics = raw_acc.values()
        self._last_phi_raw = {
            c: float(phi_sum[i]) / decisions
            for i, c in enumerate(feature_bank.components)
        }

    def _phase_changed(self, agents, last_phase, cur_phase):
        """Per-agent 0/1: did the executed phase change this decision (switches cost)."""
        return [
            bool(np.any(np.asarray(last_phase[i]) != np.asarray(cur_phase[i])))
            for i in range(len(agents))
        ]

    def _feature_matrix(self, feature_bank, agents, last_phase, cur_phase):
        changed = self._phase_changed(agents, last_phase, cur_phase)
        phi = np.stack(
            [feature_bank.features(i, changed[i]) for i in range(len(agents))]
        )
        # This decision's interval is consumed; start accumulating the next one.
        feature_bank.reset_interval()
        return phi

    # ---- rollouts ---------------------------------------------------------
    def run_train_episode(self, *, env, metric, agents, feature_bank, episode, desc):
        metric.clear()
        last_obs = env.reset()
        for agent in agents:
            agent.reset()
        raw_acc = self._raw_acc_sim if env is self.env_sim else self._raw_acc_real
        raw_acc.reset()
        feature_bank.reset_interval()
        episode_loss = []
        shaped_sum, shaped_n = 0.0, 0  # mean reward the agent ACTUALLY trains on
        i = 0
        dones = [False] * len(agents)
        pbar = tqdm(total=int(self.steps / self.action_interval), desc=desc)

        while i < self.steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])
                actions = []
                for idx, ag in enumerate(agents):
                    actions.append(
                        ag.get_action(last_obs[idx], last_phase[idx], test=False)
                    )
                actions = np.stack(actions)
                actions_prob = [
                    ag.get_action_prob(last_obs[idx], last_phase[idx])
                    for idx, ag in enumerate(agents)
                ]

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    raw_acc.step()
                    feature_bank.step_accumulate()
                proxy_rewards = np.mean(rewards_list, axis=0)
                metric.update(proxy_rewards)  # native proxy -> "rewards" column

                cur_phase = np.stack([ag.get_phase() for ag in agents])
                # Training reward: native proxy (naive) or the method's transform.
                if self.reward_transform is None:
                    train_rewards = proxy_rewards.astype(float)
                else:
                    phi = self._feature_matrix(
                        feature_bank, agents, last_phase, cur_phase
                    )
                    train_rewards = np.asarray(
                        self.reward_transform(phi), dtype=float
                    )
                shaped_sum += float(np.mean(train_rewards))
                shaped_n += 1

                for idx, ag in enumerate(agents):
                    ag.remember(
                        last_obs[idx],
                        last_phase[idx],
                        actions[idx],
                        actions_prob[idx],
                        train_rewards[idx],
                        obs[idx],
                        cur_phase[idx],
                        dones[idx],
                        f"{episode}_{i // self.action_interval}_{ag.id}",
                    )
                self.total_decision_num_sim += 1
                last_obs = obs

            if (
                self.total_decision_num_sim > self.learning_start
                and self.total_decision_num_sim % self.update_model_rate
                == self.update_model_rate - 1
            ):
                episode_loss.append(np.stack([ag.train() for ag in agents]))
            if (
                self.total_decision_num_sim > self.learning_start
                and self.total_decision_num_sim % self.update_target_rate
                == self.update_target_rate - 1
            ):
                [ag.update_target_network() for ag in agents]
            if all(dones):
                break
        pbar.close()
        mean_loss = np.mean(np.array(episode_loss)) if episode_loss else 0
        # Mean transformed reward the agent trained on this episode (= native proxy for
        # naive; the LinearReward(w) signal for the shaping methods). Logged separately
        # from the `rewards` column, which always reports the native proxy.
        self._last_train_reward = shaped_sum / shaped_n if shaped_n else 0.0
        # Raw metrics of THIS rollout for the upcoming DTL row; phi_raw is eval-only.
        self._last_raw_metrics = raw_acc.values()
        self._last_phi_raw = {}
        return mean_loss, i

    def run_eval_episode(self, *, env, metric, agents, feature_bank, desc,
                         count_budget=True):
        """Eval rollout that also accumulates the TRUE objective `R_real` + its
        per-component breakdown (summed over decisions and agents).

        `count_budget` (default True): count this real rollout against the real budget.
        Pass False for scoring-only evals (final test / naive-family transfer curve)."""
        if count_budget:
            self._real_rollouts += 1
        metric.clear()
        obs = env.reset()
        for agent in agents:
            agent.reset()
        raw_acc = self._raw_acc_sim if env is self.env_sim else self._raw_acc_real
        raw_acc.reset()
        feature_bank.reset_interval()
        i = 0
        dones = [False] * len(agents)
        phi_sum = np.zeros(len(feature_bank.components))
        pbar = tqdm(total=int(self.test_steps / self.action_interval), desc=desc)

        while i < self.test_steps:
            if i % self.action_interval == 0:
                pbar.update()
                last_phase = np.stack([ag.get_phase() for ag in agents])
                actions = np.stack(
                    [
                        ag.get_action(obs[idx], last_phase[idx], test=True)
                        for idx, ag in enumerate(agents)
                    ]
                )
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    raw_acc.step()
                    feature_bank.step_accumulate()
                metric.update(np.mean(rewards_list, axis=0))
                cur_phase = np.stack([ag.get_phase() for ag in agents])
                phi = self._feature_matrix(feature_bank, agents, last_phase, cur_phase)
                phi_sum += phi.sum(axis=0)
            if all(dones):
                break
        pbar.close()
        decisions = max(int(self.test_steps / self.action_interval), 1)
        true_reward = float(self.true_reward.reward(phi_sum[None, :])[0]) / decisions
        breakdown = self.true_reward.breakdown(phi_sum[None, :])
        breakdown = {k: v / decisions for k, v in breakdown.items()}
        self._stash_raw(raw_acc, feature_bank, phi_sum, decisions)
        return true_reward, breakdown

    # ---- logging ----------------------------------------------------------
    def _log_sim_train(self, loss, metric=None):
        """Emit one SIM_TRAIN DTL row at the current monotonic sim-episode step, then
        advance it. Used by every method's sim training so the train curve interleaves
        with real-eval rows (consistent with the base loop / other tasks). Records the
        transformed training reward (`_last_train_reward`) in the `train_reward` column."""
        self.log_metrics(
            "SIM_TRAIN", self._sim_train_step, metric or self.metric_sim, loss,
            train_reward=self._last_train_reward,
        )
        self._sim_train_step += 1

    def _w_detail(self, w):
        """Compact, TSV-safe rendering of a weight vector for the DTL `detail`
        column: `queue=1.0;delay=0.9` (nonzero components only)."""
        return ";".join(
            f"{c}={round(float(w[i]), 3)}"
            for i, c in enumerate(self.components)
            if w[i] > 1e-3
        )

    def log_metrics(self, mode, step, metric, loss, true_reward=None, breakdown=None,
                    detail="", train_reward=None):
        msg = (
            "%s step:%s, travel time:%s, q_loss:%s, rewards:%s, queue:%s, delay:%s, "
            "throughput:%s"
        )
        args = [
            mode,
            step,
            metric.real_average_travel_time(),
            loss,
            metric.rewards(),
            metric.queue(),
            metric.delay(),
            int(metric.throughput()),
        ]
        if true_reward is not None:
            msg += ", R_real:%.4f"
            args.append(true_reward)
            if breakdown:
                msg += ", components:%s"
                args.append({k: round(v, 3) for k, v in breakdown.items()})
        if self._last_raw_metrics:
            msg += ", raw_metrics:%s"
            args.append({k: round(v, 3) for k, v in self._last_raw_metrics.items()})
        if detail:
            msg += ", %s"
            args.append(detail)
        self.logger.info(msg, *args)
        self.writeLog(
            mode,
            step,
            metric.real_average_travel_time(),
            loss,
            metric.rewards(),
            metric.queue(),
            metric.delay(),
            metric.throughput(),
            true_reward if true_reward is not None else 0.0,
            breakdown,
            detail,
            train_reward,
        )

    # ---- train / test -----------------------------------------------------
    def sim_train(self, episode):
        self.on_episode_start(episode)
        mean_loss, steps_run = self.run_train_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            agents=self.agents_sim,
            feature_bank=self.feature_bank_sim,
            episode=episode,
            desc=f"SIM_TRAIN Epoch {episode}",
        )
        self._log_sim_train(mean_loss)
        self.logger.info("sim step:%s/%s", steps_run, self.steps)
        return mean_loss

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())
        for episode in range(self.sim_episodes):
            sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)
            if episode % self.save_rate == 0:
                self.save_agents(self.agents_sim, self.model_dir, e=episode)
            self.logger.info(
                "episode:%s/%s, sim_loss:%s", episode, self.sim_episodes, sim_loss
            )
            if self.test_when_train and self._should_run_real_eval(episode):
                self.train_test(episode)
        self.save_agents(self.agents_sim, self.model_dir, e=self.sim_episodes)
        self.save_agents(self.agents_sim, self.model_dir)
        self._log_real_budget()

    def _should_run_real_eval(self, episode):
        # Mirror the observation trainers: every `real_eval_interval` episodes
        # (skipping episode 0); interval 0 -> every episode.
        if self.real_eval_interval <= 0:
            return True
        return episode > 0 and episode % self.real_eval_interval == 0

    def _save_method_state(self, state):
        """Persist a small JSON of method-level results (identified/selected weights,
        BO history, ...) next to the policy weights -- reproducibility: the weights
        alone don't tell you WHAT the method inferred/selected."""
        import json

        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, "method_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=float)
        self.logger.info("Saved method state to %s", path)

    def _log_real_budget(self):
        """Report real rollouts spent vs the cap. Every method calls this at the end of
        `train()` so the DTL/run log carry an auditable budget number. Invariant: the
        count equals the number of `REAL_TRAIN` rows in the DTL (scoring evals use the
        REAL_TEST mode and are not counted)."""
        cap = self.real_episodes or 0
        self.logger.info(
            "real rollouts used: %s/%s", self._real_rollouts, cap
        )
        if cap and self._real_rollouts > cap:
            self.logger.warning(
                "real budget EXCEEDED: used %s > cap %s", self._real_rollouts, cap
            )

    def train_test(self, episode):
        # Scoring-only transfer-curve eval (naive family); NOT charged to the real
        # budget and logged as REAL_TEST (scoring), NOT REAL_TRAIN, so the count of REAL_TRAIN
        # rows stays equal to the real budget spent (see _log_real_budget).
        self.load_agents(self.agents_real, self.model_dir)
        true_reward, breakdown = self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            feature_bank=self.feature_bank_real,
            desc=f"REAL_TEST Epoch {episode}",
            count_budget=False,
        )
        self.log_metrics(
            "REAL_TEST", episode, self.metric_real, 100, true_reward, breakdown
        )
        return true_reward

    def test(self, drop_load=False):
        # Final benchmark scoring eval; NOT charged to the real budget, logged as
        # REAL_TEST (repo-wide convention), free of the real budget.
        if not drop_load:
            self.load_agents(self.agents_real, self.model_dir, e=self.sim_episodes)
        true_reward, breakdown = self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            agents=self.agents_real,
            feature_bank=self.feature_bank_real,
            desc="REAL_TEST",
            count_budget=False,
        )
        self.log_metrics(
            "REAL_TEST", 0, self.metric_real, 100, true_reward, breakdown
        )
        return self.metric_real

    def writeLog(
        self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay,
        cur_throughput, true_reward=0.0, components=None, detail="", train_reward=None,
    ):
        # Per-term R_real breakdown, TSV-safe (`queue=-0.4;emission=-0.6`).
        comp_str = (
            ";".join(f"{k}={round(float(v), 4)}" for k, v in components.items())
            if components else ""
        )
        # `rewards` is the native proxy; `train_reward` is what the agent trained on.
        train_str = "%.2f" % train_reward if train_reward is not None else ""
        # Raw benchmark metrics of the rollout this row reports (units in the
        # DTL_COLUMNS comment). Empty cell = this side's simulator can't compute the
        # metric (cityflow: all but fairness) -- distinct from a genuine 0.
        raw = self._last_raw_metrics
        raw_cols = ["%.4f" % raw[m] if m in raw else "" for m in RAW_METRIC_COLUMNS]
        # Raw UNNORMALIZED per-decision phi (eval rows only) -- the numbers behind
        # the weighted `components` column.
        phi_str = ";".join(
            f"{k}={round(float(v), 4)}" for k, v in self._last_phi_raw.items()
        )
        res = (
            self.exp_name
            + "\t" + mode
            + "\t" + str(step)
            + "\t" + "%.1f" % travel_time
            + "\t" + "%.1f" % loss
            + "\t" + "%.2f" % cur_rwd
            + "\t" + "%.2f" % cur_queue
            + "\t" + "%.2f" % cur_delay
            + "\t" + "%d" % cur_throughput
            + "\t" + train_str
            + "\t" + "%.4f" % true_reward
            + "\t" + comp_str
            + "\t" + (detail or "")
            + "\t" + "\t".join(raw_cols)
            + "\t" + phi_str
        )
        with open(self.log_file, "a") as f:
            f.write(res + "\n")
