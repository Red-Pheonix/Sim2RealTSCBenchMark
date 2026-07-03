"""morl_grid -- fixed-grid multi-policy scalarization with deploy-time selection.

The **multiple-policy** branch of multi-objective RL: scalarize the vector objective with a
SET of weight vectors, solve each as an ordinary single-objective problem (one `w` -> one
policy), and pick the policy for the (initially unknown) real preference at deployment. We
use the simplest, oldest form of this idea -- a **fixed grid** of weights -- so the honest
anchor is the early multiple-policy / set-of-weights literature, NOT the modern adaptive
coverage-set machinery:

  * Natarajan & Tadepalli, "Dynamic Preferences in Multi-Criteria Reinforcement Learning,"
    ICML 2005 -- learn a SET of policies over a range of weight vectors, store them, SELECT
    the right one when the preference is revealed, then REFINE that selected policy further.
    This is almost exactly our build-set-then-select(-then-refine) pipeline, predating OLS.
  * Roijers, Vamplew, Whiteson & Dazeley, JAIR 2013 (survey) -- the "decision-support /
    unknown-weights" scenario: keep a coverage set, choose a member on revelation. This is
    what licenses our framing (the real preference `w*` is hidden until real-eval).
  * Vamplew et al., Machine Learning 2011 -- the "set of weights -> Pareto front" evaluation
    methodology.

We deliberately use the **fixed grid** (each component corner + the uniform mix) rather than
the *adaptive* corner-weight selection of Optimistic Linear Support / Deep OLS (Mossalam,
Assael, Roijers & Whiteson 2016, arXiv:1610.02707) -- that modern adaptive coverage-set
construction is the version we do NOT reproduce. We also **select by real `R_real`** (the
preference is unknown) instead of assembling the set for known preferences. Reuses the
existing DQN/PressLight agents; spends ~|grid| real episodes on selection. (NOT the
single-network *envelope* branch, Yang/Sun/Narasimhan NeurIPS 2019 -- a conditioned
`Q(s,a,w)` method we do not implement.)

Difference from reward_inference / dynamic_reward_shaping: reward_inference *infers* `w` by
regression then trains one policy; dynamic_reward_shaping *adaptively* searches `w` by BO;
morl_grid trains a *fixed set* spanning the simplex and *picks* by real performance (it is
the non-adaptive, grid-search member of the search-then-select family).
"""

import itertools

import numpy as np

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import LinearReward


@Registry.register_trainer("sim2real_rewards_morl_grid")
class Sim2RealRewardsMORLGridTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "morl_grid"
        cfg = self.get_sim2real_config()
        self.morl_episodes = int(cfg.get("morl_episodes", 3))  # sim episodes per grid w
        # Extra sim training of the SELECTED weight after grid search (Natarajan &
        # Tadepalli 2005: once the preference is fixed, refine that policy further).
        # 0 -> reuse morl_episodes.
        self.final_episodes = int(cfg.get("final_episodes", 0))
        # Validate (and keep-best) the refine policy every this-many episodes; the
        # actual rate is max(this, ceil(refine_eps / remaining_real_budget)).
        self.refine_eval_rate = int(cfg.get("refine_eval_rate", 2))
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.weight_grid = self._build_grid(cfg.get("weight_grid"))

    def build_reward_transform(self, feature_bank):
        return None

    def _build_grid(self, grid_cfg):
        """Weight vectors spanning the simplex (or an explicit list of `{component: w}`
        dicts from the config). Default layout, in order:
          1. the UNIFORM mix (first, so it's never the point dropped by a tight real
             budget -- it is usually the strongest single candidate);
          2. each single-component corner;
          3. every pairwise 50/50 mix (C(n,2)).
        With the 5 sim-core components -> 1 + 5 + 10 = 16 grid points. Only
        SIM-COMPUTABLE components are used -- a grid point on a sim-unavailable term
        (e.g. emission in cityflow) trains on an all-zero reward (degenerate)."""
        if grid_cfg:
            return [self.feature_bank_sim.weight_vector(g) for g in grid_cfg]
        comps = self.feature_bank_sim.available_components()
        grid = [self.feature_bank_sim.weight_vector({c: 1.0 / len(comps) for c in comps})]
        grid += [self.feature_bank_sim.weight_vector({c: 1.0}) for c in comps]
        grid += [
            self.feature_bank_sim.weight_vector({a: 0.5, b: 0.5})
            for a, b in itertools.combinations(comps, 2)
        ]
        return grid

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())
        # Snapshot the (pretrained) weights so every grid policy reset warm-starts from
        # here instead of random init -- parity with naive.
        self._capture_reset_base()

        # Grid selection spends one real rollout per grid point; reserve the rest of the
        # real budget for the refine validations below. With real_episodes=100 and a
        # 16-point grid this never truncates, but warn (not silently drop) if it does.
        budget = self.real_episodes or len(self.weight_grid)
        if budget < len(self.weight_grid):
            self.logger.warning(
                "MORLGrid: real budget %s < grid size %s; dropping %s grid point(s) "
                "(uniform mix is first, so it is retained)",
                budget, len(self.weight_grid), len(self.weight_grid) - budget,
            )
        grid = self.weight_grid[:budget]
        best_r, best_dir, best_w = -np.inf, None, None
        self.logger.info("MORLGrid: %s grid policies", len(grid))
        for gi, w in enumerate(grid):
            self._reset_sim_policy()
            self.reward_transform = LinearReward(
                w, self.components, self.reward_scale, norm=self.component_norm
            )
            for ep in range(self.morl_episodes):
                self.on_episode_start(ep)
                loss, _ = self.run_train_episode(
                    env=self.env_sim,
                    metric=self.metric_sim,
                    agents=self.agents_sim,
                    feature_bank=self.feature_bank_sim,
                    episode=ep,
                    desc=f"SIM_TRAIN MORLGrid[w{gi}] Epoch {ep}",
                )
                self._log_sim_train(loss)
            # Deploy this grid policy once in real; keep the best by R_real. Log a
            # standard REAL_TRAIN row to the DTL (grid index as step, weights in detail).
            self.save_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)
            r_real, breakdown = self.run_eval_episode(
                env=self.env_real,
                metric=self.metric_real,
                agents=self.agents_real,
                feature_bank=self.feature_bank_real,
                desc=f"REAL_TRAIN MORLGrid[w{gi}] select",
            )
            self.log_metrics(
                "REAL_TRAIN", gi + 1, self.metric_real, 100, r_real, breakdown,
                f"grid;w={self._w_detail(w)}",
            )
            if r_real > best_r:
                best_r = r_real
                best_w = w
                best_dir = self.model_dir + f"_morl_grid_best_{gi}"
                self.save_agents(self.agents_sim, best_dir)
        # --- Refine the selected weight: with the preference now fixed, train its
        # policy longer (Natarajan & Tadepalli 2005 -- refine the chosen policy once the
        # preference is determined), KEEPING THE BEST checkpoint by real R_real. The
        # selected grid policy is the floor, so refine can never ship worse than what
        # the grid search picked (parity with reward_inference's phase-3 keep-best). ---
        if best_dir is not None:
            # Start refine from a CLEAN agent (fresh replay buffer + decision counter)
            # loaded with the best grid's weights. Reusing the last grid's agent would
            # train on its STALE, gridlocked buffer and collapse the selected policy.
            self._reset_sim_policy(warm_start=False)
            self.load_agents(self.agents_sim, best_dir)
            self.reward_transform = LinearReward(
                best_w, self.components, self.reward_scale, norm=self.component_norm
            )
            refine_episodes = self.final_episodes or self.morl_episodes
            # Cap refine validations by the remaining real budget (grid already spent
            # len(grid)); validate every `refine_eval_rate` episodes.
            val_budget = refine_episodes
            if self.real_episodes:
                val_budget = min(val_budget, max(1, self.real_episodes - len(grid)))
            # CEILING division (floor would validate every episode when
            # refine_episodes < 2*val_budget, e.g. with refine_eval_rate=1); keeps
            # grid + refine validations within the real cap.
            eval_rate = max(
                self.refine_eval_rate, -(-refine_episodes // max(1, val_budget))
            )
            refine_best_dir = self.model_dir + "_morl_refine_best"

            def _validate_refine(step, detail):
                self.save_agents(self.agents_sim, self.model_dir)
                self.load_agents(self.agents_real, self.model_dir)
                r, bd = self.run_eval_episode(
                    env=self.env_real, metric=self.metric_real, agents=self.agents_real,
                    feature_bank=self.feature_bank_real,
                    desc=f"REAL_TRAIN MORLGrid-Refine[{detail}]",
                )
                self.log_metrics(
                    "REAL_TRAIN", step, self.metric_real, 100, r, bd,
                    f"refine;w={self._w_detail(best_w)};{detail}",
                )
                return float(r)

            for ep in range(refine_episodes):
                self.on_episode_start(ep)
                loss, _ = self.run_train_episode(
                    env=self.env_sim,
                    metric=self.metric_sim,
                    agents=self.agents_sim,
                    feature_bank=self.feature_bank_sim,
                    episode=ep,
                    desc=f"SIM_TRAIN MORLGrid-Refine Epoch {ep}",
                )
                self._log_sim_train(loss)
                if (ep + 1) % eval_rate == 0 or ep == refine_episodes - 1:
                    r_ref = _validate_refine(len(grid) + 1 + ep, f"ckpt{ep + 1}")
                    if r_ref > best_r:
                        best_r = r_ref
                        self.save_agents(self.agents_sim, refine_best_dir)
                        best_dir = refine_best_dir  # promote refined checkpoint
            # Promote the best real-validated policy (selected grid floor or a refined
            # checkpoint that beat it).
            self.load_agents(self.agents_sim, best_dir)
            self.save_agents(self.agents_sim, self.model_dir, e=self.sim_episodes)
            self.save_agents(self.agents_sim, self.model_dir)
        self.logger.info("MORLGrid selected policy with R_real=%.4f", best_r)
        if best_w is not None:
            self._save_method_state({
                "selected_w": {
                    c: float(best_w[i]) for i, c in enumerate(self.components)
                },
                "grid_size": len(grid),
                "best_R_real": float(best_r),
            })
        self._log_real_budget()
