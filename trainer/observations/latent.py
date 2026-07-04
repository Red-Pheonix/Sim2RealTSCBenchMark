"""Generic latent-observation trainer.

Two stages:

Stage 1 -- the model owns the encoder weights and decides per-episode whether
to roll out under the default sim config or a freshly DR-sampled one (via
`prepare_latent_train_episode`). The trainer drives the rollouts and feeds
each per-intersection obs into the model via `save_batch`. After all episodes,
the model trains its encoders and saves them to a per-method cache dir.

Stage 2 -- fresh latent-input PressLight agents (Q-net resized to
`latent_dim + phase`) train on the default sim config. The model's encoder is
hooked into each agent's `get_ob` so env-side obs flow through the latent
transparently; the agent stays raw-obs-blind.
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from common.registry import Registry
from .base import BaseObservationTrainer


@Registry.register_trainer("sim2real_observations_latent")
class LatentObservationTrainer(BaseObservationTrainer):
    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)

        method = self.sim2real_config["latent_obs_method"]
        model_cls = Registry.mapping["sim2real_model_mapping"][method]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_cls(
            config=self.sim2real_config.get(method, {}),
            num_intersections=len(self.agents_sim),
            obs_dims=[ag.ob_generator.ob_length for ag in self.agents_sim],
            device=device,
        )
        self.enc_dec_cache_dir = os.path.join(
            "pretrained",
            method,
            Registry.mapping["command_mapping"]["setting"].param["network"],
        )

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def train_encoders(self):
        paths = self.model.cache_paths(self.enc_dec_cache_dir)
        if not self.model.force_retrain and all(os.path.exists(p) for p in paths):
            self.logger.info("Loading cached latent encoders from %s", self.enc_dec_cache_dir)
            self.model.load(paths)
            self._save_run_encoders()
            return

        self.load_agents(self.agents_sim, self.pretrained_model_dir())

        for episode_id in tqdm(
            range(self.model.n_pretrain_episodes),
            desc="Stage 1: collecting episodes",
        ):
            self.collect_latent_train_episode(episode_id)

        self.model.train()

        os.makedirs(self.enc_dec_cache_dir, exist_ok=True)
        # ATOMIC cache write: concurrent runs of the same (method, network) share this
        # cache; write to unique tmp files then rename so a concurrent reader never
        # loads a half-written checkpoint.
        tmp_paths = [f"{p}.tmp{os.getpid()}" for p in paths]
        self.model.save(tmp_paths)
        for tmp, dst in zip(tmp_paths, paths):
            os.replace(tmp, dst)
        self._save_run_encoders()

    def _save_run_encoders(self):
        """Persist the encoders THIS run used into the run's model_dir (alongside the
        policy checkpoints), so the run is reproducible on its own even if the shared
        per-method cache (`pretrained/<method>/<network>/`) is later overwritten."""
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model.cache_paths(self.model_dir))

    def collect_latent_train_episode(self, episode_id):
        self.model.prepare_latent_train_episode(episode_id, self)
        obs = self.reset_episode(self.env_sim, self.world_sim, self.agents_sim)
        step = 0
        actions = np.zeros(len(self.agents_sim), dtype=int)
        dones = [False] * len(self.agents_sim)
        while step < self.steps:
            if step % self.action_interval == 0:
                for i in range(len(self.agents_sim)):
                    self.model.save_batch(i, obs[i], episode_id)
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = np.stack([
                    ag.get_action(obs[j], phases[j], test=True)
                    for j, ag in enumerate(self.agents_sim)
                ])
            obs, _, dones, _ = self.env_sim.step(actions.flatten())
            step += 1
            if all(dones):
                break

    # ------------------------------------------------------------------
    # Stage 2 setup
    # ------------------------------------------------------------------

    def build_latent_agents(self):
        """Resize each agent's Q-net to (latent_dim + phase) and reinit from
        scratch. The raw ob_generator stays untouched -- the encoder consumes
        the env's raw obs upstream and the agent only ever sees latent vectors."""
        for agents in (self.agents_sim, self.agents_real):
            for i, ag in enumerate(agents):
                phase_extra = ag.ob_length - ag.ob_generator.ob_length
                ag.rebuild_model(self.model.latent_dim(i) + phase_extra)

    def attach_latent_encoders(self):
        """Wrap each agent's get_ob so env.reset / env.step return latent obs
        instead of raw lane counts. Single hook -- downstream get_action,
        remember, _batchwise all flow through encoded obs transparently."""
        for agents in (self.agents_sim, self.agents_real):
            for i, ag in enumerate(agents):
                ag.get_ob = self.make_encoded_get_ob(ag.get_ob, i)
        self._latent_ready = True

    def _ensure_latent_stack(self):
        """Rebuild the latent stack from the RUN's saved encoders (fall back to the
        shared cache). No-op when train() already set the stack up in this process.
        The encoder cache is setting-independent, so this works for cross-setting
        replay (the run-scoped path won't exist -> shared cache is used)."""
        if not getattr(self, "_latent_ready", False):
            run_paths = self.model.cache_paths(self.model_dir)
            if all(os.path.exists(p) for p in run_paths):
                self.logger.info("Loading run-scoped latent encoders from %s", self.model_dir)
                self.model.load(run_paths)
            else:
                self.logger.info(
                    "Loading cached latent encoders from %s", self.enc_dec_cache_dir
                )
                self.model.load(self.model.cache_paths(self.enc_dec_cache_dir))
            self.build_latent_agents()
            self.attach_latent_encoders()

    def prepare_eval(self):
        # Checkpoint-replay (train-once-eval-many) needs the encoder stack too.
        self._ensure_latent_stack()

    def test(self, drop_load=False):
        """Eval-only entry (train_model: False): rebuild the latent stack before
        the base test. No-op when train() already set the stack up in this process."""
        self._ensure_latent_stack()
        return super().test(drop_load)

    def make_encoded_get_ob(self, raw_get_ob, intersection_id):
        def encoded():
            raw = raw_get_ob()
            return self.model.encode(intersection_id, torch.from_numpy(raw)).cpu().numpy()
        return encoded

    # ------------------------------------------------------------------
    # Stage 2 RL loop
    # ------------------------------------------------------------------

    def run_train_episode(
        self,
        *,
        env,
        metric,
        world,
        agents,
        episode,
        total_decision_num,
        desc,
    ):
        metric.clear()
        last_obs = self.reset_episode(env, world, agents)

        episode_loss = []
        flush = 0
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

                actions_prob = []
                for idx, ag in enumerate(agents):
                    actions_prob.append(
                        ag.get_action_prob(last_obs[idx], last_phase[idx])
                    )

                rewards_list = []
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)
                metric.update(rewards)

                cur_phase = np.stack([ag.get_phase() for ag in agents])
                for idx, ag in enumerate(agents):
                    ag.remember(
                        last_obs[idx],
                        last_phase[idx],
                        actions[idx],
                        actions_prob[idx],
                        rewards[idx],
                        obs[idx],
                        cur_phase[idx],
                        dones[idx],
                        f"{episode}_{i // self.action_interval}_{ag.id}",
                    )

                flush += 1
                if flush == self.buffer_size - 1:
                    flush = 0

                total_decision_num += 1
                last_obs = obs

            if (
                total_decision_num > self.learning_start
                and total_decision_num % self.update_model_rate
                == self.update_model_rate - 1
            ):
                cur_loss_q = np.stack([ag.train() for ag in agents])
                episode_loss.append(cur_loss_q)

            if (
                total_decision_num > self.learning_start
                and total_decision_num % self.update_target_rate
                == self.update_target_rate - 1
            ):
                [ag.update_target_network() for ag in agents]

            if all(dones):
                break

        pbar.close()

        mean_loss = np.mean(np.array(episode_loss)) if episode_loss else 0
        return total_decision_num, mean_loss, i

    def sim_train(self, episode):
        self.set_replay(
            self.env_sim,
            f"sim_episode_{episode}.txt",
            episode % self.save_rate == 0,
        )
        self.total_decision_num_sim, mean_loss, steps_run = self.run_train_episode(
            env=self.env_sim,
            metric=self.metric_sim,
            world=self.world_sim,
            agents=self.agents_sim,
            episode=episode,
            total_decision_num=self.total_decision_num_sim,
            desc=f"Sim Training Epoch {episode}",
        )
        self.log_metrics("SIM_TRAIN", episode, self.metric_sim, mean_loss)
        self.logger.info("sim step:%s/%s", steps_run, self.steps)
        return mean_loss

    def should_run_real_eval(self, episode):
        return (
            self.real_eval_interval > 0
            and episode > 0
            and episode % self.real_eval_interval == 0
        )

    def train_test(self, episode):
        self.load_agents(self.agents_real, self.model_dir)
        self.run_eval_episode(
            env=self.env_real,
            metric=self.metric_real,
            world=self.world_real,
            agents=self.agents_real,
            desc=f"Real Eval Epoch {episode}",
        )
        self.log_metrics("REAL_TEST", episode, self.metric_real, 100)
        return self.metric_real.real_average_travel_time()

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def train(self):
        # ---- Stage 1: encoders ----
        self.train_encoders()

        # ---- Stage 2: fresh latent-space agents on default sim config ----
        self.build_latent_agents()
        self.attach_latent_encoders()
        self.apply_default_sim_domain()

        for episode in range(self.episodes):
            sim_loss = self.sim_train(episode)
            self.save_agents(self.agents_sim, self.model_dir)

            if episode % self.save_rate == 0:
                self.save_agents(self.agents_sim, self.model_dir, e=episode)

            self.logger.info(
                "episode:%s/%s, sim_loss:%s",
                episode,
                self.episodes,
                sim_loss,
            )

            if self.should_run_real_eval(episode):
                self.train_test(episode)

        # Final e-tagged checkpoint so test()/eval-only reloads (e=self.episodes) work.
        self.save_agents(self.agents_sim, self.model_dir, e=self.episodes)
        self.save_agents(self.agents_sim, self.model_dir)
