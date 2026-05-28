"""LUSR trainer for observation-side sim2real experiments.

Stage 1: collect observations across N DR configs with the pretrained
PressLight, train one cycle-consistent disentangled VAE per intersection on
the collected per-domain data.

Stage 2: instantiate fresh PressLight agents with `ob_length = content_dim`,
train them from scratch on the frozen encoder's content latent. Real eval
runs the same pipeline against the real env's observation transforms.

Model code (Encoder / Decoder / DisentangledVAE / LUSRDataset / LUSR) is
lifted from the test.py sandbox; see notes/lusr.md for the deeper writeup
and references to the original paper / repo (github.com/KarlXing/LUSR).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.registry import Registry
from .base import BaseObservationTrainer


# ---------------------------------------------------------------------------
# LUSR model + dataset (lifted from test.py)
# ---------------------------------------------------------------------------

def reparameterize(mu, logsigma):
    std = torch.exp(0.5 * logsigma)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, obs_dim, content_dim, class_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.linear_mu = nn.Linear(hidden, content_dim)
        self.linear_logsigma = nn.Linear(hidden, content_dim)
        self.linear_classcode = nn.Linear(hidden, class_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.linear_mu(h), self.linear_logsigma(h), self.linear_classcode(h)

    def get_feature(self, x):
        mu, _, _ = self.forward(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class DisentangledVAE(nn.Module):
    def __init__(self, obs_dim, content_dim=8, class_dim=4, hidden=128):
        super().__init__()
        self.encoder = Encoder(obs_dim, content_dim, class_dim, hidden)
        self.decoder = Decoder(content_dim + class_dim, obs_dim, hidden)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        recon = self.decoder(torch.cat([contentcode, classcode], dim=1))
        return mu, logsigma, classcode, recon


def vae_loss(x, mu, logsigma, recon_x, beta):
    recon = F.mse_loss(x, recon_x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl = kl / torch.numel(x)
    return recon + beta * kl


def forward_loss_batched(x, model, beta):
    # x: (D, B, obs_dim). One encoder/decoder pass over D*B samples; the class-code
    # shuffle stays within each domain via a per-domain randperm.
    D, B, obs_dim = x.shape
    flat_x = x.reshape(D * B, obs_dim)
    mu, logsigma, classcode = model.encoder(flat_x)
    contentcode = reparameterize(mu, logsigma)

    class_dim = classcode.shape[-1]
    perms = torch.argsort(torch.rand(D, B, device=x.device), dim=1)
    shuffled_classcode = torch.gather(
        classcode.view(D, B, class_dim),
        1,
        perms.unsqueeze(-1).expand(-1, -1, class_dim),
    ).view(D * B, class_dim)

    recon_shuffled = model.decoder(torch.cat([contentcode, shuffled_classcode], dim=1))
    recon_original = model.decoder(torch.cat([contentcode, classcode], dim=1))
    return (
        vae_loss(flat_x, mu, logsigma, recon_shuffled, beta)
        + vae_loss(flat_x, mu, logsigma, recon_original, beta)
    )


def backward_loss(x, model, device):
    mu, _, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0], device=device)]
    rand_content = torch.randn_like(mu, device=device)
    recon1 = model.decoder(torch.cat([rand_content, classcode], dim=1)).detach()
    recon2 = model.decoder(torch.cat([rand_content, shuffled_classcode], dim=1)).detach()
    cycle_mu1, cycle_logsigma1, _ = model.encoder(recon1)
    cycle_mu2, cycle_logsigma2, _ = model.encoder(recon2)
    cycle_content1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_content2 = reparameterize(cycle_mu2, cycle_logsigma2)
    return F.l1_loss(cycle_content1, cycle_content2)


class LUSRDataset:
    """In-memory per-domain observation buffer. All inputs are torch.Tensor of shape (N, obs_dim)."""

    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.chunks = {}

    def add(self, obs, domain_id):
        self.chunks.setdefault(domain_id, []).append(obs.detach().cpu().float())

    @property
    def num_domains(self):
        return len(self.chunks)

    def stack(self):
        return [
            torch.cat(self.chunks[k], dim=0) for k in sorted(self.chunks.keys())
        ]

    def iter_batches(self, batch_size, device=None):
        """Yield (D, B, obs_dim) tensors. All batches are prebuilt up front."""
        per_domain = self.stack()
        D = len(per_domain)
        min_len = min(t.shape[0] for t in per_domain)
        num_batches = min_len // batch_size
        used_len = num_batches * batch_size
        batches = (
            torch.stack(
                [t[torch.randperm(t.shape[0])[:used_len]] for t in per_domain],
                dim=0,
            )
            .view(D, num_batches, batch_size, -1)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        if device is not None:
            batches = batches.to(device, non_blocking=True)
        for b in range(num_batches):
            yield batches[b]

    def as_dict(self):
        return {k: torch.cat(v, dim=0) for k, v in self.chunks.items()}


class LUSR:
    """DisentangledVAE + per-domain dataset wrapper. One instance per intersection."""

    def __init__(self, obs_dim, content_dim=8, class_dim=8, hidden=128, device=None):
        self.obs_dim = obs_dim
        self.content_dim = content_dim
        self.class_dim = class_dim
        self.hidden = hidden
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vae = DisentangledVAE(obs_dim, content_dim, class_dim, hidden).to(
            self.device
        )
        self.dataset = LUSRDataset(obs_dim)

    def save_batch(self, obs, domain_id):
        self.dataset.add(obs, domain_id)

    def train(
        self,
        epochs=50,
        batch_size=64,
        beta=10.0,
        bloss_coef=1.0,
        lr=1e-4,
        verbose=True,
    ):
        if self.dataset.num_domains < 2:
            raise ValueError(
                "LUSR needs samples from at least two domains; "
                f"got {self.dataset.num_domains}."
            )
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        pbar = tqdm(range(epochs), desc="LUSR train", disable=not verbose)
        for _ in pbar:
            self.vae.train()
            floss_sum = 0.0
            bloss_sum = 0.0
            steps = 0

            for batch_by_domain in self.dataset.iter_batches(batch_size, device=self.device):
                optimizer.zero_grad()

                floss = forward_loss_batched(batch_by_domain, self.vae, beta)

                flat = batch_by_domain.reshape(-1, self.obs_dim)
                bloss = backward_loss(flat, self.vae, self.device)

                (floss + bloss_coef * bloss).backward()
                optimizer.step()

                floss_sum += floss.item()
                bloss_sum += bloss.item()
                steps += 1

            if steps > 0:
                pbar.set_postfix(
                    floss=f"{floss_sum / steps:.4f}",
                    bloss=f"{bloss_sum / steps:.4f}",
                )

    def encode(self, obs):
        """obs: (N, obs_dim) tensor -> (N, content_dim) tensor."""
        self.vae.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            return self.vae.encoder.get_feature(obs)

    def save(self, path, include_dataset=False):
        payload = {
            "vae": self.vae.state_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "content_dim": self.content_dim,
                "class_dim": self.class_dim,
                "hidden": self.hidden,
            },
        }
        if include_dataset:
            payload["dataset"] = self.dataset.as_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device=None):
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        lusr = cls(
            obs_dim=cfg["obs_dim"],
            content_dim=cfg["content_dim"],
            class_dim=cfg["class_dim"],
            hidden=cfg["hidden"],
            device=device,
        )
        lusr.vae.load_state_dict(ckpt["vae"])
        lusr.vae.to(lusr.device)
        if "dataset" in ckpt:
            for k, v in ckpt["dataset"].items():
                lusr.dataset.chunks[k] = [v]
        return lusr


# ---------------------------------------------------------------------------
# Trainer (skeleton; mirrors ObservationDomainRandomizationTrainer)
# ---------------------------------------------------------------------------

@Registry.register_trainer("sim2real_observations_lusr")
class ObservationLUSRTrainer(BaseObservationTrainer):
    """Two-stage LUSR sim2real trainer.

    Stage 1: collect observations across N pre-sampled DR configs using the
    pretrained PressLight policy; train one LUSR (cycle-consistent VAE) per
    intersection on the collected per-domain data.

    Stage 2: instantiate fresh PressLight agents with `ob_length = content_dim`,
    train them from scratch on the frozen encoder's content latent. Real eval
    runs the same pipeline against the real env's observation transforms.
    """

    def __init__(self, logger, gpu=0, cpu=False, name="sim2real_observations"):
        super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)
        lusr_cfg = self.sim2real_config.get("lusr", {})
        self.content_dim = lusr_cfg.get("content_dim", 8)
        self.class_dim = lusr_cfg.get("class_dim", 4)
        self.hidden = lusr_cfg.get("hidden", 128)
        self.pretrain_episodes = lusr_cfg.get("pretrain_episodes", 50)
        self.lusr_epochs = lusr_cfg.get("epochs", 50)
        self.lusr_batch_size = lusr_cfg.get("batch_size", 64)
        self.lusr_beta = lusr_cfg.get("beta", 10.0)
        self.lusr_bloss_coef = lusr_cfg.get("bloss_coef", 1.0)
        self.lusr_lr = lusr_cfg.get("lr", 1e-4)
        self.force_retrain = lusr_cfg.get("force_retrain", False)
        self.lusrs = None

    def enc_dec_cache_dir(self):
        return os.path.join(
            "pretrained",
            "lusr",
            Registry.mapping["command_mapping"]["setting"].param["network"],
        )

    def enc_dec_cache_paths(self, n):
        return [
            os.path.join(self.enc_dec_cache_dir(), f"intersection_{i}.pt")
            for i in range(n)
        ]

    def build_latent_agents(self):
        """Resize each agent's Q-net to (content_dim + phase) and reinit from
        scratch. The raw ob_generator stays untouched -- the encoder consumes
        the env's raw obs upstream and the agent only ever sees content_dim
        vectors."""
        for agents in (self.agents_sim, self.agents_real):
            for i, ag in enumerate(agents):
                phase_extra = ag.ob_length - ag.ob_generator.ob_length
                ag.rebuild_model(self.lusrs[i].content_dim + phase_extra)

    def attach_latent_encoders(self):
        """Wrap each agent's get_ob so env.reset / env.step return content-latent
        obs instead of raw lane counts. Single hook -- downstream get_action,
        remember, _batchwise all flow through encoded obs transparently."""
        for agents in (self.agents_sim, self.agents_real):
            for i, ag in enumerate(agents):
                ag.get_ob = self.make_encoded_get_ob(ag.get_ob, self.lusrs[i])

    @staticmethod
    def make_encoded_get_ob(raw_get_ob, lusr):
        def encoded():
            raw = raw_get_ob()
            return lusr.encode(torch.from_numpy(raw)).cpu().numpy()
        return encoded

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

    def latent_sim_train(self, domain_id):
        """Roll out the pretrained policy for one episode (self.steps env steps)
        and save per-intersection observations into self.lusrs[i] tagged with
        domain_id. Domain 0 uses the default sim config (no obs transforms);
        subsequent domains sample a fresh DR config."""
        if domain_id == 0:
            self.apply_default_sim_domain()
        else:
            self.apply_new_sim_domain()

        obs = self.reset_episode(self.env_sim, self.world_sim, self.agents_sim)
        step = 0
        actions = np.zeros(len(self.agents_sim), dtype=int)
        dones = [False] * len(self.agents_sim)
        while step < self.steps:
            if step % self.action_interval == 0:
                for i, lusr in enumerate(self.lusrs):
                    lusr.save_batch(
                        torch.from_numpy(obs[i].astype(np.float32)),
                        domain_id=domain_id,
                    )
                phases = np.stack([ag.get_phase() for ag in self.agents_sim])
                actions = np.stack([
                    ag.get_action(obs[j], phases[j], test=True)
                    for j, ag in enumerate(self.agents_sim)
                ])
            obs, _, dones, _ = self.env_sim.step(actions.flatten())
            step += 1
            if all(dones):
                break

    def train_lusrs(self):
        """Stage 1 wrapper: load the pretrained policy, instantiate one LUSR
        per intersection, collect observations across N DR configs (one episode
        each), then train each disentangled VAE."""
        cache_paths = self.enc_dec_cache_paths(len(self.agents_sim))
        if not self.force_retrain and all(os.path.exists(p) for p in cache_paths):
            self.logger.info("Loading cached LUSR encoders from %s", self.enc_dec_cache_dir())
            self.lusrs = [LUSR.load(p) for p in cache_paths]
            return

        self.load_agents(self.agents_sim, self.pretrained_model_dir())

        self.lusrs = [
            LUSR(
                obs_dim=ag.ob_generator.ob_length,
                content_dim=self.content_dim,
                class_dim=self.class_dim,
                hidden=self.hidden,
            )
            for ag in self.agents_sim
        ]

        for domain_id in tqdm(range(self.pretrain_episodes), desc="Stage 1: collecting domains"):
            self.latent_sim_train(domain_id)

        os.makedirs(self.enc_dec_cache_dir(), exist_ok=True)
        for i, lusr in enumerate(self.lusrs):
            self.logger.info("Training LUSR encoder %d/%d", i + 1, len(self.lusrs))
            lusr.train(
                epochs=self.lusr_epochs,
                batch_size=self.lusr_batch_size,
                beta=self.lusr_beta,
                bloss_coef=self.lusr_bloss_coef,
                lr=self.lusr_lr,
            )
            lusr.save(cache_paths[i])

    def train(self):
        # ---- Stage 1: collect per-domain data + train per-intersection encoders ----
        self.train_lusrs()

        # ---- Stage 2: fresh latent-space agents on default sim config ----
        self.build_latent_agents()
        self.attach_latent_encoders()
        self.apply_default_sim_domain()

        # Note: load_pretrained intentionally NOT honored at stage 2. Paper
        # recipe trains the Q-net from scratch on the content latent.

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

        self.save_agents(self.agents_sim, self.model_dir)

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
        self.log_metrics("TEST_REAL", episode, self.metric_real, 100)
        return self.metric_real.real_average_travel_time()
