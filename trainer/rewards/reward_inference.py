"""reward_inference -- recover the hidden objective `w*` by regression, then optimise.

Real reward feedback is expensive and scarce: each real deployment returns only one
episode-level scalar, far too little to train a policy in real. But it IS enough to
*identify* the objective. Each real deployment `k` gives one linear equation in the
unknown cost weights:

    R_real^(k) = - Φ^(k) · w*          (Φ^(k) = accumulated feature/cost vector)

`w*` has only `d` parameters, so `K ≈ d` real episodes recover it by (ridge) regression
-- the linear-reward / feature-matching view of reward learning (Abbeel & Ng, ICML 2004;
linear-in-features reward, Ng & Russell, ICML 2000). Too few scalars to train a policy,
enough to identify the objective. We then train the final policy in cheap sim on
`LinearReward(ŵ)`. (Framing: the specified sim proxy is an *observation* of the true
objective, cf. Inverse Reward Design, Hadfield-Menell et al. 2017 -- but we keep the
point ridge estimate, not IRD's risk-averse posterior planning; that robustness angle
lives in `reward_random`.)

This is the **passive / fixed-design** member of the reward-learning-from-rollout-ratings
family (Active Reward Learning, Daniel, Viering, Metz, Kroemer & Peters, RSS 2014, shares
the data model: a scalar reward per deployed rollout). Two deliberate departures: (1) we
DROP their active acquisition -- probes are a FIXED open-loop design rather than queries
chosen online to be maximally informative (the closed-loop / active variant is the
separate `dynamic_reward_shaping` method, which also *optimizes the outcome* instead of
identifying `w*`);
(2) point ridge estimate with a nonneg clip (costs have `w >= 0`), not a GP reward model.

Probe design: rather than query arbitrary rollouts, we choose probe policies that induce
*diverse* feature profiles (one short probe per component, each on a unit-weight
`LinearReward(e_c)`, cycled `probe_repeats` times with fresh inits) so the `K x d` system
is better-conditioned -- an open-loop experimental design for identifiability. Each probe
is deployed once in real -> one `(Φ, R_real)` row.

Real budget (shared 300-ep pool, real <= 100): probes + 1 warm-start floor eval +
keep-best validations during phase-3 final training. Every real rollout that shapes the
deployed policy (probes identify ŵ; validations pick the checkpoint) is counted; only the
final benchmark scoring eval is free. See notes/reward_gap_fix_plan.md (Task 7).
"""

import os

import numpy as np

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import LinearReward, _norm_vector


@Registry.register_trainer("sim2real_rewards_reward_inference")
class Sim2RealRewardsInferenceTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "reward_inference"
        cfg = self.get_sim2real_config()
        # Sim episodes per probe policy (kept small -- a probe only needs to induce a
        # distinct feature profile, not converge).
        self.probe_episodes = int(cfg.get("probe_episodes", 3))
        # Probe-only learning_start. The global learning_start (1000) exceeds a probe's
        # whole decision budget (probe_episodes * 360), so at the default the probe DQN
        # never calls train() -- probes stay random-init and their per-component reward
        # is unused. Lower it just for probes so they actually learn a distinct policy;
        # phase-3 final training keeps the normal learning_start.
        self.probe_learning_start = int(cfg.get("probe_learning_start", 200))
        # Which components to probe. Default = SIM-AVAILABLE only: a probe policy
        # trains in sim, so probing a sim-unobservable term (emission in cityflow)
        # just trains on a zero reward. The objective's real-only terms are still
        # SCORED in real; they're simply not actionable from sim (see train()).
        self.probe_components = cfg.get(
            "probe_components", self.feature_bank_sim.available_components()
        )
        # Repeats of the probe set (each repeat uses fresh random inits, so identical
        # single-component probes still induce SLIGHTLY different feature profiles ->
        # more rows for the ridge and a marginally better-conditioned system). With ~5
        # sim components and probe_repeats 3 -> 15 probe rows.
        self.probe_repeats = int(cfg.get("probe_repeats", 3))
        self.ridge_lambda = float(cfg.get("ridge_lambda", 1e-2))
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        # Phase-3 final-training length (sim episodes). Explicit knob: `sim_episodes` is
        # now the whole 300-ep pool reference, NOT phase 3's length, so we must not
        # reuse it here (that would blow the budget). Default 155 keeps
        # probes(≈45) + final(155) = 200 sim, leaving 100 real for probes+validations.
        self.final_episodes = int(cfg.get("final_episodes", 155))
        # Number of real validations during final training -> keep the best checkpoint
        # (parity with morl_grid/DRS; the warm-start is the floor). Guards against
        # shipping a final policy that gridlocked on a weak actionable reward. The
        # actual validation count is capped by the real budget in train() (real =
        # probes + 1 warm-start floor + validations <= real_episodes).
        self.final_evals = int(cfg.get("final_evals", 84))
        self.identified_w = None

    # naive transform during probe/final is set explicitly in train(); the base
    # __init__ default (None) is overridden per phase.
    def build_reward_transform(self, feature_bank):
        return None

    def _collect_real(self):
        """Deploy the current sim policy in real for one episode; return the
        accumulated cost vector `Φ` and the scalar `R_real`. Counts against the real
        budget (a probe is a real rollout that shapes ŵ, hence the deployed policy)."""
        self._real_rollouts += 1
        self.save_agents(self.agents_sim, self.model_dir)
        self.load_agents(self.agents_real, self.model_dir)
        self.metric_real.clear()  # populate metric so the probe can log a standard DTL row
        phi_sum = np.zeros(len(self.components))
        obs = self.env_real.reset()
        for ag in self.agents_real:
            ag.reset()
        i = 0
        dones = [False] * len(self.agents_real)
        while i < self.test_steps:
            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents_real])
                actions = np.stack(
                    [
                        ag.get_action(obs[idx], last_phase[idx], test=True)
                        for idx, ag in enumerate(self.agents_real)
                    ]
                )
                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env_real.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                self.metric_real.update(np.mean(rewards_list, axis=0))
                cur_phase = np.stack([ag.get_phase() for ag in self.agents_real])
                phi = self._feature_matrix(
                    self.feature_bank_real, self.agents_real, last_phase, cur_phase
                )
                phi_sum += phi.sum(axis=0)
            if all(dones):
                break
        r_real = float(self.true_reward.reward(phi_sum[None, :])[0])
        return phi_sum, r_real

    def _train_sim(self, episodes, transform, tag):
        """Train the (fresh) sim policy for `episodes` on `transform`."""
        self.reward_transform = transform
        for ep in range(episodes):
            self.on_episode_start(ep)
            loss, _ = self.run_train_episode(
                env=self.env_sim,
                metric=self.metric_sim,
                agents=self.agents_sim,
                feature_bank=self.feature_bank_sim,
                episode=ep,
                desc=f"TRAIN_SIM {tag} Epoch {ep}",
            )
            self._log_sim_train(loss)
        self.save_agents(self.agents_sim, self.model_dir)

    def _ridge(self, phi_rows, y):
        """ŵ = argmin ||Φ ŵ - (-y)||^2 + λ||ŵ||^2, then clip to nonneg (costs).

        Φ is NORMALIZED (raw φ / component_norm) before the solve so ŵ lives in the
        SAME space as LinearReward/TrueReward, which score `-( (φ/norm) · w )`. Without
        this, `_collect_real` returns raw φ while the target R_real is built from
        normalized φ, so the fit would give ŵ_i ≈ w*_i / n_i -- and Phase 3 divides by
        the norm AGAIN, distorting relative weights by a factor n_i per component. The
        normalization also makes `ridge_lambda` penalize every component on one scale.
        """
        Phi = np.asarray(phi_rows, dtype=float) / _norm_vector(
            self.components, self.component_norm
        )
        target = -np.asarray(y, dtype=float)  # -R_real = (Φ/norm) · w*
        d = Phi.shape[1]
        A = Phi.T @ Phi + self.ridge_lambda * np.eye(d)
        w = np.linalg.solve(A, Phi.T @ target)
        return np.clip(w, 0.0, None)

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())
        # Snapshot the (pretrained) weights so every probe/final reset warm-starts from
        # here instead of random init -- parity with naive.
        self._capture_reset_base()

        # --- Phase 1: diverse probes -> (Φ, R_real) rows (spends real budget) ---
        # Probe list = sim-available components cycled `probe_repeats` times. Reserve 1
        # real rollout for the phase-3 warm-start floor eval, so probes take at most
        # real_episodes - 1 of the budget.
        phi_rows, y = [], []
        probe_list = list(self.probe_components) * self.probe_repeats
        probe_cap = max(1, (self.real_episodes or len(probe_list)) - 1)
        probe_list = probe_list[:probe_cap]
        self.logger.info(
            "RewardInference: %s probes (components %s x %s repeats, capped at %s by real budget)",
            len(probe_list), self.probe_components, self.probe_repeats, probe_cap,
        )
        for pi, c in enumerate(probe_list):
            w = self.feature_bank_sim.weight_vector({c: 1.0})
            # Probes need DIVERSE feature profiles -> independent random inits, NOT a
            # shared warm-start (which collapses them to the same policy). The final
            # scored policy below DOES warm-start (parity with naive).
            # NOTE: this is a KNOWN-WEAK spot. Both random-init (probes gridlock to the
            # same saturated state) and warm-start (probes don't move -> identical Φ)
            # produce near-collinear Φ, so single-component identification is poorly
            # conditioned -- compounded by the sim components (queue/delay/waiting/
            # pressure) being correlated congestion proxies that aren't separately
            # identifiable. The aggregate (congestion) direction IS captured, which is
            # what drives the final policy. A genuinely diverse probe design is open work.
            self._reset_sim_policy(warm_start=False)
            # Probe-only learning_start: the global one (1000) exceeds a probe's whole
            # decision budget, so without this the probe DQN never trains and every
            # probe stays random-init (its per-component reward unused). Restore after.
            ls_saved = self.learning_start
            self.learning_start = self.probe_learning_start
            try:
                self._train_sim(
                    self.probe_episodes,
                    LinearReward(w, self.components, self.reward_scale, norm=self.component_norm),
                    f"Probe[{c}#{pi}]",
                )
            finally:
                self.learning_start = ls_saved
            phi, r_real = self._collect_real()
            phi_rows.append(phi)
            y.append(r_real)
            # Log a standard TEST_REAL row (probe index as step). The ridge keeps the
            # raw `r_real`/Φ; the logged R_real is normalized per-decision to match the
            # scale of the other methods' TEST_REAL rows.
            decisions = max(int(self.test_steps / self.action_interval), 1)
            breakdown = {
                k: v / decisions
                for k, v in self.true_reward.breakdown(phi[None, :]).items()
            }
            self.log_metrics(
                "TEST_REAL", len(phi_rows), self.metric_real, 100,
                r_real / decisions, breakdown, f"probe[{c}]",
            )

        # --- Phase 2: ridge-solve ŵ (probe Φ comes from REAL, so all terms are
        # observed and identifiable here, including sim-unavailable ones). ---
        self.identified_w = self._ridge(phi_rows, y)
        self.logger.info(
            "RewardInference identified ŵ: %s",
            {c: round(float(self.identified_w[i]), 4) for i, c in enumerate(self.components)},
        )

        # Split ŵ: what SIM can act on vs real-only terms (identified but inert in sim
        # training -- the irreducible part of the gap; only correlated available proxies
        # / shaping could touch them).
        avail = self.feature_bank_sim.available_mask()
        actionable_w = self.identified_w * avail
        unobservable = {
            c: round(float(self.identified_w[i]), 4)
            for i, c in enumerate(self.components)
            if avail[i] == 0.0 and self.identified_w[i] > 1e-6
        }
        if unobservable:
            self.logger.info(
                "RewardInference: identified but SIM-UNOBSERVABLE (irreducible gap): %s", unobservable
            )

        # Renormalize the actionable weights to a TRAINABLE scale. After masking sim-blind
        # terms (e.g. emission), the surviving actionable weights can be tiny (~0.05) -> a
        # near-zero reward the DQN can't learn from, so it diverges with no signal to
        # recover (this is exactly the emission_heavy "break"). Rescale so the strongest
        # actionable weight is 1.0: the RATIO among actionable terms (which is all that
        # drives the policy) is preserved, and scoring still uses the true weights. If
        # there is no actionable weight at all, leave it (keep-best deploys the warm-start).
        amax = float(np.max(actionable_w)) if actionable_w.size else 0.0
        if amax > 1e-9:
            actionable_w = actionable_w / amax
            self.logger.info(
                "RewardInference: renormalized actionable ŵ (max->1): %s",
                {c: round(float(actionable_w[i]), 4)
                 for i, c in enumerate(self.components) if actionable_w[i] > 1e-3},
            )

        # --- Phase 3: train the final policy in sim on R̂ = LinearReward(actionable ŵ),
        # but KEEP THE BEST checkpoint by real R_real (parity with morl_grid/DRS, which
        # deploy a real-validated SAVED policy instead of the last training state).
        # Without this, a final that gridlocks on a weak actionable reward (e.g. queue-only
        # on an emission-dominated objective) would be shipped as-is; here the warm-start
        # is the floor, so reward_inference can never deploy worse than naive. Probes are
        # the REAL-budget cost; the deployed policy still gets the full sim budget. ---
        final_episodes = self.final_episodes
        # Cap validations by the remaining real budget: real = probes + 1 warm-start
        # floor + validations <= real_episodes. Reserve the probes already spent + the
        # warm-start, then spread the rest over the final training.
        val_budget = self.final_evals
        if self.real_episodes:
            val_budget = min(val_budget, max(1, self.real_episodes - len(phi_rows) - 1))
        # CEILING division: floor gives rate 1 whenever final_episodes < 2*val_budget
        # (e.g. 155//84 = 1 -> validate EVERY episode -> 155 rollouts, blowing the cap).
        # Ceiling keeps validations <= val_budget so probes + 1 + validations <= real cap.
        eval_rate = max(1, -(-final_episodes // max(1, val_budget)))
        self._reset_sim_policy()
        self.reward_transform = LinearReward(
            actionable_w, self.components, self.reward_scale, norm=self.component_norm
        )
        best_dir = self.model_dir + "_ri_best"

        def _validate_real(step, detail):
            self.save_agents(self.agents_sim, self.model_dir)
            self.load_agents(self.agents_real, self.model_dir)
            r, bd = self.run_eval_episode(
                env=self.env_real, metric=self.metric_real, agents=self.agents_real,
                feature_bank=self.feature_bank_real,
                desc=f"TEST_REAL RewardInference-Final[{detail}]",
            )
            self.log_metrics("TEST_REAL", step, self.metric_real, 100, r, bd, f"final;{detail}")
            return float(r)

        # Warm-start (== naive policy) is the floor.
        best_r = _validate_real(0, "warmstart")
        self.save_agents(self.agents_sim, best_dir)
        for ep in range(final_episodes):
            self.on_episode_start(ep)
            loss, _ = self.run_train_episode(
                env=self.env_sim, metric=self.metric_sim, agents=self.agents_sim,
                feature_bank=self.feature_bank_sim, episode=ep,
                desc="TRAIN_SIM RewardInference-Final",
            )
            self._log_sim_train(loss)
            if (ep + 1) % eval_rate == 0 or ep == final_episodes - 1:
                r = _validate_real(ep + 1, "ckpt")
                if r > best_r:
                    best_r = r
                    self.save_agents(self.agents_sim, best_dir)
        # Promote the best real-validated checkpoint.
        self.load_agents(self.agents_sim, best_dir)
        self.logger.info("RewardInference-Final: deploying best checkpoint R_real=%.4f", best_r)
        self.save_agents(self.agents_sim, self.model_dir, e=self.sim_episodes)
        self.save_agents(self.agents_sim, self.model_dir)
        self._log_real_budget()
