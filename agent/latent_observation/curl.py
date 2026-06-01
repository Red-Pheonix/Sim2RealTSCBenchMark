"""CURL latent-observation model.

CURL (Contrastive Unsupervised Representations for Reinforcement Learning) was
originally proposed by Laskin, Srinivas, Abbeel (2020) for sample-efficient RL
from pixels. We follow the LUSR paper's (Xing et al. 2021) practice of
re-casting CURL as a two-stage representation-learning baseline for
domain adaptation: pre-train the encoder on multi-domain observations, freeze
it, then train RL on the frozen latent.

Reference code: https://github.com/MishaLaskin/curl
    - `curl_sac.py`: CURL class, `update_cpc`, `compute_logits`
    - `utils.py`: `random_crop` (image augmentation we replace below)

Key adaptations for our setting:
- Pre-train then freeze (instead of joint with RL as in the original).
- Multi-domain data collection (matching the LUSR paper's shared dataset for
  baselines: episode 0 = default sim config, later episodes = DR-sampled).
- Vector-obs augmentation: Gaussian noise + per-feature dropout, in place of
  the original random-crop on pixels.

LUSR paper observes that CURL embeddings tend to cluster by domain, hurting
transfer (see Section "Domain Adaptation After Training"). Expect this method
to underperform LUSR/ATC on real-domain eval -- the failure is informative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.registry import Registry
from .base import LatentObservationModel


# ---------------------------------------------------------------------------
# Encoder + augmentation
# ---------------------------------------------------------------------------

class CURLEncoder(nn.Module):
    """MLP encoder. Output is the latent passed to the bilinear scorer / RL agent.

    Original CURL uses a 4-layer CNN for pixel input; we use an MLP because
    obs is a small lane-count vector (~24 dims). We keep the reference's output
    head: LayerNorm + tanh on the latent (curl_sac.py PixelEncoder.forward:
    `out = tanh(ln(fc(h)))`), which bounds the latent and stabilizes the
    bilinear InfoNCE score."""

    def __init__(self, obs_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, x):
        return torch.tanh(self.ln(self.net(x)))


def vector_augment(x, noise_std, mask_prob):
    """Compose Gaussian noise + per-feature dropout to produce one augmented
    view. Called twice on the same source obs to get anchor and positive views.

    Original CURL uses random crops of input images; for state vectors there is
    no spatial structure to crop, so we substitute the two most common vector
    augmentations from the contrastive-RL-on-states literature."""
    x = x + noise_std * torch.randn_like(x)
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


# ---------------------------------------------------------------------------
# CURLCore: one instance per intersection (encoder + EMA target + W + buffer)
# ---------------------------------------------------------------------------

class CURLCore:
    """Wraps a query encoder + EMA target encoder + bilinear scorer W + flat
    obs buffer for a single intersection.

    Training mirrors `update_cpc` in
    https://github.com/MishaLaskin/curl/blob/master/curl_sac.py:
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        logits = self.CURL.compute_logits(z_a, z_pos)
        loss = self.cross_entropy_loss(logits, labels)
    with `logits = z_a @ (W @ z_pos.T)` and `labels = arange(B)` (diagonal)."""

    def __init__(self, obs_dim, latent_dim=8, hidden=128, device=None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.query_encoder = CURLEncoder(obs_dim, latent_dim, hidden).to(self.device)
        self.target_encoder = CURLEncoder(obs_dim, latent_dim, hidden).to(self.device)
        # Initialize target as a copy of query; target is never trained directly,
        # only EMA-updated.
        self.target_encoder.load_state_dict(self.query_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Bilinear similarity matrix. The reference uses random init
        # (curl_sac.py: `self.W = nn.Parameter(torch.rand(z_dim, z_dim))`),
        # so W is a learnable metric rather than starting at plain dot product.
        self.W = nn.Parameter(torch.rand(latent_dim, latent_dim, device=self.device))

        # Flat in-memory buffer (no domain tagging needed for CURL).
        self.buffer = []

    def add(self, obs):
        """obs: (obs_dim,) numpy or (1, obs_dim) tensor -> stored as (obs_dim,) float CPU."""
        if isinstance(obs, torch.Tensor):
            t = obs.detach().cpu().float().reshape(-1)
        else:
            t = torch.from_numpy(obs.astype("float32")).reshape(-1)
        self.buffer.append(t)

    def stack(self):
        return torch.stack(self.buffer, dim=0)  # (N, obs_dim) on CPU

    def train(
        self,
        epochs=100,
        batch_size=64,
        lr=1e-4,
        encoder_tau=0.005,
        aug_noise_std=0.1,
        aug_mask_prob=0.1,
        verbose=True,
    ):
        """InfoNCE loop. Each step:
            1. Sample a batch x of obs.
            2. Produce two augmented views x_q, x_k (anchor, positive).
            3. z_q = query_encoder(x_q); z_k = target_encoder(x_k) (no grad).
            4. logits = z_q @ (W @ z_k.T)  -> (B, B)
            5. CE loss with diagonal labels (InfoNCE in CE form).
            6. SGD on query_encoder + W; EMA update on target_encoder.
        """
        all_obs = self.stack().to(self.device)
        N = all_obs.shape[0]
        num_batches = N // batch_size
        if num_batches == 0:
            raise ValueError(
                f"CURL needs at least batch_size={batch_size} obs; got {N}."
            )

        optimizer = torch.optim.Adam(
            list(self.query_encoder.parameters()) + [self.W], lr=lr
        )

        pbar = tqdm(range(epochs), desc="CURL train", disable=not verbose)
        for _ in pbar:
            self.query_encoder.train()
            perm = torch.randperm(N, device=self.device)
            loss_sum = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                x = all_obs[idx]

                # Two independent augmented views of the same source batch.
                x_q = vector_augment(x, aug_noise_std, aug_mask_prob)
                x_k = vector_augment(x, aug_noise_std, aug_mask_prob)

                z_q = self.query_encoder(x_q)              # (B, latent_dim)
                with torch.no_grad():
                    z_k = self.target_encoder(x_k)         # (B, latent_dim), no grad

                # Bilinear similarity. Subtract per-row max for numerical
                # stability (same trick used in curl_sac.py).
                logits = z_q @ (self.W @ z_k.t())          # (B, B)
                logits = logits - logits.max(dim=1, keepdim=True).values.detach()

                labels = torch.arange(logits.shape[0], device=self.device)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # EMA update: target = tau * query + (1 - tau) * target.
                # Mirrors `utils.soft_update_params` in the reference repo.
                with torch.no_grad():
                    for p_t, p_q in zip(
                        self.target_encoder.parameters(),
                        self.query_encoder.parameters(),
                    ):
                        p_t.data.mul_(1.0 - encoder_tau).add_(p_q.data, alpha=encoder_tau)

                loss_sum += loss.item()

            if verbose:
                pbar.set_postfix(loss=f"{loss_sum / num_batches:.4f}")

    def encode(self, obs):
        """obs: (N, obs_dim) tensor -> (N, latent_dim) tensor (no augmentation,
        no gradient). Used by the trainer's monkey-patched get_ob during
        stage-2 RL training and eval."""
        self.query_encoder.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            return self.query_encoder(obs)

    def save(self, path):
        # Save only the query encoder (used at inference) + config to rebuild
        # the architecture. W and target encoder are training-only state.
        torch.save(
            {
                "encoder": self.query_encoder.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "latent_dim": self.latent_dim,
                    "hidden": self.hidden,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        core = cls(
            obs_dim=cfg["obs_dim"],
            latent_dim=cfg["latent_dim"],
            hidden=cfg["hidden"],
            device=device,
        )
        core.query_encoder.load_state_dict(ckpt["encoder"])
        core.query_encoder.to(core.device)
        return core


# ---------------------------------------------------------------------------
# Trainer-facing wrapper
# ---------------------------------------------------------------------------

@Registry.register_sim2real_model("curl")
class CURLModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.cores = [
            CURLCore(
                obs_dim=obs_dims[i],
                latent_dim=config.get("latent_dim", 8),
                hidden=config.get("hidden", 128),
                device=device,
            )
            for i in range(num_intersections)
        ]
        self._training_kwargs = {
            "epochs": config.get("epochs", 100),
            "batch_size": config.get("batch_size", 64),
            "lr": config.get("lr", 1e-4),
            "encoder_tau": config.get("encoder_tau", 0.005),
            "aug_noise_std": config.get("aug_noise_std", 0.1),
            "aug_mask_prob": config.get("aug_mask_prob", 0.1),
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 100)

    def prepare_latent_train_episode(self, episode_id, trainer):
        # Multi-domain collection, matching the LUSR paper's shared dataset
        # across baselines. Episode 0 = source domain (default sim config);
        # later episodes = freshly DR-sampled domains.
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        # CURL has no domain concept: obs go into a flat buffer; the
        # augmentation supplies the only contrastive variation.
        self.cores[intersection_id].add(obs)

    def train(self):
        for core in self.cores:
            core.train(**self._training_kwargs)

    def latent_dim(self, intersection_id):
        return self.cores[intersection_id].latent_dim

    def encode(self, intersection_id, obs_tensor):
        return self.cores[intersection_id].encode(obs_tensor)

    def save(self, paths):
        for core, path in zip(self.cores, paths):
            core.save(path)

    def load(self, paths):
        self.cores = [CURLCore.load(p, device=self.device) for p in paths]
