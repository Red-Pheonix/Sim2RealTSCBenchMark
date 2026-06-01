"""ATC latent-observation model.

Augmented Temporal Contrast, introduced in Stooke, Lee, Abbeel, Laskin (2021),
"Decoupling Representation Learning from Reinforcement Learning"
(https://arxiv.org/abs/2009.08319).

Reference code: https://github.com/astooke/rlpyt
    - `rlpyt/ul/algos/ul_for_rl/atc.py`: ATC algorithm
    - `rlpyt/ul/models/ul/atc_models.py`: contrastive head with bilinear scorer

Difference from CURL:
- Positive pair is the *future* state `s_{t+k}` (reference default k=1; we use
  k=3 by default), not an augmentation of the same state. Augmentation is
  applied to BOTH anchor s_t and positive s_{t+k}.
- The anchor (online) branch has a residual predictor MLP before the bilinear
  scorer; the positive (momentum) branch does not. This asymmetric predictor is
  ATC's signature head (see atc_models.py ContrastModel).
- Rest of the recipe is the same: bilinear similarity, InfoNCE / cross-entropy
  on the (B, B) logits matrix, EMA target encoder.

Why this is a more transfer-friendly contrastive method than CURL: ATC's
encoder must preserve information about temporal dynamics, which tend to be
domain-invariant (queue evolution, vehicle arrivals) even when individual obs
features are corrupted by domain shift. CURL only learns augmentation
invariance, which gives a weaker pressure to ignore domain-specific cues.

Adaptations for our setting:
- Vector-obs augmentation (Gaussian noise + per-feature dropout) replaces the
  original random crop + intensity jitter on pixel images.
- Pre-train then freeze (matches our `LatentObservationTrainer` two-stage shape).
- Per-intersection encoders, multi-domain stage-1 collection.

Note: ATC pairs (s_t, s_{t+k}) must come from the SAME rollout; we cannot pair
the last obs of episode i with the first obs of episode i+1 (different env
states + different DR configs). We tag each obs with its (episode_id, step)
and only pair within an episode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.registry import Registry
from .base import LatentObservationModel


# ---------------------------------------------------------------------------
# Encoder + augmentation (same as CURL adaptation)
# ---------------------------------------------------------------------------

class ATCEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def vector_augment(x, noise_std, mask_prob):
    """Gaussian noise + per-feature dropout. See agent/latent_observation/curl.py
    for the rationale (random-crop substitute for vector obs)."""
    x = x + noise_std * torch.randn_like(x)
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


# ---------------------------------------------------------------------------
# ATCCore: per-intersection encoder + EMA target + W + temporally-ordered buffer
# ---------------------------------------------------------------------------

class ATCCore:
    """Per-intersection ATC components.

    Buffer is a dict {episode_id: list[Tensor]} so we can sample temporal
    pairs (obs at step i, obs at step i+k) from the same rollout. Mixing
    episodes would pair across different DR configs (and would be wrong)."""

    def __init__(self, obs_dim, latent_dim=8, hidden=128, anchor_hidden=512, device=None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.anchor_hidden = anchor_hidden
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.query_encoder = ATCEncoder(obs_dim, latent_dim, hidden).to(self.device)
        self.target_encoder = ATCEncoder(obs_dim, latent_dim, hidden).to(self.device)
        self.target_encoder.load_state_dict(self.query_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Residual predictor MLP on the anchor (online) branch only -- the
        # positive (momentum) branch has none. This asymmetric predictor is
        # ATC's defining head (atc_models.py ContrastModel: anchor_hidden_sizes
        # defaults to 512, so it is active by default).
        self.anchor_mlp = nn.Sequential(
            nn.Linear(latent_dim, anchor_hidden), nn.ReLU(),
            nn.Linear(anchor_hidden, latent_dim),
        ).to(self.device)

        # Bilinear scorer. Reference uses nn.Linear(latent, latent, bias=False)
        # (random kaiming init), NOT identity.
        self.W = nn.Linear(latent_dim, latent_dim, bias=False).to(self.device)

        # Per-episode ordered obs sequences. Tensors are CPU until batched.
        self.episodes = {}

    def add(self, obs, episode_id):
        if isinstance(obs, torch.Tensor):
            t = obs.detach().cpu().float().reshape(-1)
        else:
            t = torch.from_numpy(obs.astype("float32")).reshape(-1)
        self.episodes.setdefault(episode_id, []).append(t)

    def build_pair_pool(self, k):
        """Materialize (anchor, positive) pairs from every episode where
        positive = anchor shifted forward k steps within the same episode.
        Returns two stacked tensors: anchors (M, obs_dim), positives (M, obs_dim)."""
        anchors = []
        positives = []
        for ep_id, ep in self.episodes.items():
            if len(ep) <= k:
                continue
            ep_tensor = torch.stack(ep, dim=0)  # (T, obs_dim)
            anchors.append(ep_tensor[:-k])      # (T-k, obs_dim)
            positives.append(ep_tensor[k:])     # (T-k, obs_dim)
        if not anchors:
            raise ValueError(
                f"ATC needs episodes longer than k={k} steps; none available."
            )
        return torch.cat(anchors, dim=0), torch.cat(positives, dim=0)

    def train(
        self,
        epochs=100,
        batch_size=64,
        lr=1e-4,
        encoder_tau=0.01,
        aug_noise_std=0.1,
        aug_mask_prob=0.1,
        temporal_k=3,
        verbose=True,
    ):
        """For each step:
            1. Sample a batch of (anchor, positive) temporal pairs.
            2. Augment both anchor and positive independently.
            3. z_q = query_encoder(aug(anchor)); z_k = target_encoder(aug(positive)).
            4. logits = z_q @ (W @ z_k.T); CE with diagonal labels.
            5. SGD on query_encoder + W; EMA update on target_encoder.
        """
        anchors, positives = self.build_pair_pool(temporal_k)
        anchors = anchors.to(self.device)
        positives = positives.to(self.device)
        M = anchors.shape[0]
        num_batches = M // batch_size
        if num_batches == 0:
            raise ValueError(
                f"ATC needs at least batch_size={batch_size} temporal pairs; got {M}."
            )

        optimizer = torch.optim.Adam(
            list(self.query_encoder.parameters())
            + list(self.anchor_mlp.parameters())
            + list(self.W.parameters()),
            lr=lr,
        )

        pbar = tqdm(range(epochs), desc="ATC train", disable=not verbose)
        for _ in pbar:
            self.query_encoder.train()
            perm = torch.randperm(M, device=self.device)
            loss_sum = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                a = anchors[idx]
                p = positives[idx]

                # Independent augmentation on each side.
                a_aug = vector_augment(a, aug_noise_std, aug_mask_prob)
                p_aug = vector_augment(p, aug_noise_std, aug_mask_prob)

                z_q = self.query_encoder(a_aug)
                with torch.no_grad():
                    z_k = self.target_encoder(p_aug)

                # Anchor branch: residual predictor, then bilinear W; positive
                # branch is the bare momentum code (no predictor, no W).
                # logits = (W (z_q + mlp(z_q))) @ z_k.T   -> (B, B)
                anchor = z_q + self.anchor_mlp(z_q)
                logits = self.W(anchor) @ z_k.t()
                logits = logits - logits.max(dim=1, keepdim=True).values.detach()

                labels = torch.arange(logits.shape[0], device=self.device)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        self.query_encoder.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            return self.query_encoder(obs)

    def save(self, path):
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

@Registry.register_sim2real_model("atc")
class ATCModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.cores = [
            ATCCore(
                obs_dim=obs_dims[i],
                latent_dim=config.get("latent_dim", 8),
                hidden=config.get("hidden", 128),
                anchor_hidden=config.get("anchor_hidden", 512),
                device=device,
            )
            for i in range(num_intersections)
        ]
        self._training_kwargs = {
            "epochs": config.get("epochs", 100),
            "batch_size": config.get("batch_size", 64),
            "lr": config.get("lr", 1e-4),
            "encoder_tau": config.get("encoder_tau", 0.01),
            "aug_noise_std": config.get("aug_noise_std", 0.1),
            "aug_mask_prob": config.get("aug_mask_prob", 0.1),
            "temporal_k": config.get("temporal_k", 3),
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 100)

    def prepare_latent_train_episode(self, episode_id, trainer):
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        # Tag with episode_id so we only pair temporally within a single rollout.
        self.cores[intersection_id].add(obs, episode_id)

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
        self.cores = [ATCCore.load(p, device=self.device) for p in paths]
