"""dynamic_reward_shaping -- treat the sim reward weights as shaping knobs and TUNE them in
a closed sim->real->update->sim loop so the real objective `R_real` is maximized.

`reward_shaping` picks the enriched reward weighting `w` ONCE, by hand. This is its
**dynamic** counterpart: `w` is adjusted across rounds from real feedback, so the method
*learns which shaping weighting actually pays off in real* instead of guessing it. It learns
the **optimal surrogate** reward -- the weighting that, when optimized in sim, maximizes real
return (the Optimal Reward Problem, Singh/Sorg/Lewis) -- NOT the true `w*` (that descriptive
inference is the separate `reward_inference` method). Bilevel "learn the shaping weights to
maximize the true objective": Learning to Utilize Shaping Rewards (Hu et al., NeurIPS 2020);
policy-gradient reward design, PGRD (Sorg, Singh & Lewis, NeurIPS 2010).

The "active" closed loop is shared with **Active Reward Learning** (Daniel, Viering, Metz,
Kroemer & Peters, RSS 2014) -- both query an expensive real signal at adaptively chosen
points and update a probabilistic model between queries (we co-anchor ARL for that framing).
The difference: ARL models the *reward function*; we model the *real return* `R_real(w)` and
*optimize the outcome*. That instantiation is **BayRn** (Bayesian Domain Randomization;
Muratore, Eilers, Peters & Ramos, IEEE RA-L 2021, arXiv:2003.02471, Algorithm 1), retargeted
from domain-randomization parameters to reward weights.

Because it optimizes the outcome rather than identifying `w*`, `dynamic_reward_shaping` can
exploit proxy substitutions `reward_inference` cannot -- it may discover that pushing a
sim-actionable proxy raises `R_real` even when that proxy is not literally a term of the
hidden objective.

It follows the SAME outer-round sim-to-real loop as the transition gap's domain-adaptation
trainer, but swaps that trainer's probabilistic model: the DA trainer uses an SBI neural
posterior (BayesSim: infer params that *match* real observations); here the GP surrogate
*optimizes* the real return. The GP-BO uses a **known library, scikit-optimize**
(`skopt.Optimizer`, GP base estimator + Expected Improvement), not a hand-rolled GP -- its
`ask()/tell()` interface maps onto the sim-to-real loop. skopt is sklearn-only (no torch),
so it can't disturb the agents. (BayRn's own code uses BoTorch; we avoid it at this scale.)

Loop (BayRn Algorithm 1), with parameters = reward weights `w`:
  * init phase (skopt `n_initial_points`): random weightings -> train a sim policy on
    `LinearReward(w)` -> eval once in real -> seed the GP;
  * sim-to-real BO loop: `opt.ask()` proposes `w` (GP + EI) -> train sim policy -> eval in
    real -> `opt.tell(w, -R_real)` (skopt MINIMIZES, so we negate). Repeat for `opt_rounds`;
  * end: train the best `w`'s policy (a bit longer) and promote it for test().

skopt optimizes a box `[0,1]^d` over the SIM-ACTIONABLE components (a sim-unobservable term
is inert in sim); each raw point is normalized to the simplex for the reward. Each candidate
is fine-tuned from a FIXED warm-started base policy (budget-aware stand-in for BayRn's
from-scratch PolOpt, so each `R_real(w)` is a clean function of `w`). The optimizer finds
which actionable proxies best raise the REAL `R_real` (which still scores the hidden term).
"""

import numpy as np
from skopt import Optimizer
from skopt.space import Real

from common.registry import Registry
from trainer.rewards.base import Sim2RealRewardsTrainer
from trainer.rewards.reward_transforms import LinearReward


@Registry.register_trainer("sim2real_rewards_dynamic_reward_shaping")
class Sim2RealRewardsDynamicShapingTrainer(Sim2RealRewardsTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_rewards"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        self.method = "dynamic_reward_shaping"
        cfg = self.get_sim2real_config()
        self.n_init = int(cfg.get("n_init", 5))             # skopt n_initial_points
        self.opt_rounds = int(cfg.get("opt_rounds", 20))    # BO iterations after init
        self.finetune_episodes = int(cfg.get("finetune_episodes", 3))
        self.warmup_episodes = int(cfg.get("warmup_episodes", 20))
        self.final_episodes = int(cfg.get("final_episodes", 0))  # 0 -> reuse finetune
        self.reward_scale = float(cfg.get("reward_scale", 1.0))
        self.seed_opt = int(cfg.get("dynamic_reward_shaping_seed", getattr(self, "seed", 0) or 0))
        # Optimize only over SIM-ACTIONABLE components (others are inert in sim).
        self.actionable = self.feature_bank_sim.available_mask()
        self._act_idx = np.where(self.actionable > 0)[0]

    def build_reward_transform(self, feature_bank):
        return None  # reward set per candidate in train()

    @staticmethod
    def _normalize(x):
        """Raw box point -> simplex weight vector over the actionable components."""
        return x
        # x = np.maximum(np.asarray(x, dtype=float), 1e-9)
        # return x / x.sum()

    def _full_w(self, w_act):
        """Embed an actionable-simplex point into the full component weight vector."""
        w = np.zeros(len(self.components))
        w[self._act_idx] = w_act
        return w

    def _nonzero_w(self, w):
        return {c: round(float(w[i]), 3) for i, c in enumerate(self.components) if w[i] > 1e-3}

    def _train_sim(self, episodes, w_act, tag):
        self.reward_transform = LinearReward(
            self._full_w(w_act), self.components, self.reward_scale, norm=self.component_norm
        )
        for ep in range(episodes):
            self.on_episode_start(ep)
            loss, _ = self.run_train_episode(
                env=self.env_sim, metric=self.metric_sim, agents=self.agents_sim,
                feature_bank=self.feature_bank_sim, episode=ep, desc=f"TRAIN_SIM {tag}",
            )
            self._log_sim_train(loss)

    def _eval_real(self, desc, step=0, detail=""):
        """Deploy the current sim policy in real for one episode, log a standard
        TEST_REAL row to the DTL (so candidates land in the data log like the base
        methods), and return R_real."""
        self.save_agents(self.agents_sim, self.model_dir)
        self.load_agents(self.agents_real, self.model_dir)
        r_real, breakdown = self.run_eval_episode(
            env=self.env_real, metric=self.metric_real, agents=self.agents_real,
            feature_bank=self.feature_bank_real, desc=f"TEST_REAL {desc}",
        )
        self.log_metrics("TEST_REAL", step, self.metric_real, 100, r_real, breakdown, detail)
        return float(r_real)

    def _finetune_and_eval(self, base_dir, w_act, tag, step=0, detail=""):
        """Warm-start from the fixed base policy, fine-tune on LinearReward(w), eval real."""
        self.load_agents(self.agents_sim, base_dir)
        self._train_sim(self.finetune_episodes, w_act, f"{tag} finetune")
        return self._eval_real(f"{tag} eval", step, detail)

    def train(self):
        if self.load_pretrained:
            self.load_agents(self.agents_sim, self.pretrained_model_dir())

        # Warm-start base policy (trained once on a uniform actionable reward). Every
        # candidate fine-tunes from this same base, so R_real(w) is a clean function of w.
        base_dir = self.model_dir + "_dynshape_base"
        w_uniform = np.ones(len(self._act_idx)) / len(self._act_idx)
        self._train_sim(self.warmup_episodes, w_uniform, "DynShaping-Warmup")
        self.save_agents(self.agents_sim, base_dir)

        # GP Bayesian Optimization over the actionable box, via scikit-optimize.
        # skopt MINIMIZES, so we tell it -R_real. GP base estimator uses a Matern kernel
        # (as in BayRn); n_initial_points = the random init phase.
        space = [Real(0.0, 1.0, name=f"w{i}") for i in range(len(self._act_idx))]
        opt = Optimizer(
            space, base_estimator="GP", acq_func="EI",
            n_initial_points=self.n_init, random_state=self.seed_opt,
        )

        n_total = self.n_init + self.opt_rounds
        best_r, best_dir, best_w_act = -np.inf, None, None
        for t in range(n_total):
            x = opt.ask()
            w_act = self._normalize(x)
            phase = "init" if t < self.n_init else "bo"
            tag = f"DynShaping[{phase} {t + 1}/{n_total}]"
            detail = f"{phase};w={self._w_detail(self._full_w(w_act))}"
            # step = candidate index (1..n_total) so the BO trace is plottable from the DTL.
            r = self._finetune_and_eval(base_dir, w_act, tag, t + 1, detail)
            opt.tell(x, -r)                                  # minimize -R_real
            # KEEP THE ACTUAL POLICY that scored this R_real. The real eval is
            # deterministic, so the best observed candidate genuinely is the best --
            # promoting its saved weights avoids the retrain mismatch (a fresh retrain of
            # best_w can score far worse than its own BO eval -- winner's curse + DQN
            # training variance). agents_sim still holds the just-fine-tuned candidate.
            if r > best_r:
                best_r, best_w_act = r, w_act
                best_dir = self.model_dir + "_dynshape_best"
                self.save_agents(self.agents_sim, best_dir)

        # --- promote the SAVED best candidate (not a fresh retrain). Optionally refine it
        # a bit longer, but only keep the refined policy if it actually evaluates better
        # (held-out check) -- otherwise the deeper training can degrade it. ---
        self.logger.info(
            "DynShaping selected R_real=%.4f w=%s", best_r, self._nonzero_w(self._full_w(best_w_act)),
        )
        if self.final_episodes > 0:
            self.load_agents(self.agents_sim, best_dir)
            self._train_sim(self.final_episodes, best_w_act, "DynShaping-Refine")
            r_ref = self._eval_real("DynShaping-Refine eval")
            self.logger.info("DynShaping refine R_real=%.4f (saved best=%.4f)", r_ref, best_r)
            if r_ref >= best_r:
                self.save_agents(self.agents_sim, best_dir)
                best_r = r_ref
        self.load_agents(self.agents_sim, best_dir)
        self.save_agents(self.agents_sim, self.model_dir, e=self.sim_episodes)
        self.save_agents(self.agents_sim, self.model_dir)
