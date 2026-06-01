"""VAE-Embedding latent-observation model.

Named "VAE-Embedding" in the LUSR paper (Xing et al. 2021) as one of its
baselines:

    "the latent embedding of variational autoencoder (VAE) can be used as
    an internal latent state representation in RL. We call this method as
    VAE-Embedding."

The LUSR paper notes:

    "The architecture of VAE-Embedding could be considered as a special case
    of DARLA that replaces beta-VAE with VAE and avoids the usage of DAE."

i.e. a plain VAE trained on multi-domain obs; the encoder's mean (mu) is used
as the RL state representation. No DAE pretraining, no disentanglement loss
beyond the standard KL.

Reference code for the underlying VAE arrangement used in LUSR-paper
experiments: https://github.com/KarlXing/LUSR (see model.py for the VAE
building blocks; VAE-Embedding is the simpler ablation of the same code).

Adaptations for our setting:
- MLP encoder/decoder over a (~24-dim) lane-count vector instead of a CNN
  over a 96x96 pixel image.
- Multi-domain stage-1 collection matches the LUSR paper's shared dataset
  practice (episode 0 default, later episodes DR-sampled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common.registry import Registry
from .base import LatentObservationModel


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


class VAEEncoder(nn.Module):
    """MLP encoder -> (mu, logvar) of the latent Gaussian."""

    def __init__(self, obs_dim, latent_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """MLP decoder z -> reconstructed obs."""

    def __init__(self, latent_dim, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


def vae_loss(x, recon_x, mu, logvar, beta=1.0):
    """Standard VAE loss: per-element MSE reconstruction + per-element KL.
    Both terms are normalized per element so the magnitudes are comparable
    regardless of obs_dim / batch size. beta=1.0 corresponds to vanilla VAE
    (i.e. VAE-Embedding rather than DARLA's beta-VAE)."""
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / torch.numel(x)
    return recon + beta * kl, recon, kl


class VAEEmbeddingCore:
    """Single-intersection VAE wrapper. Holds the encoder + decoder + flat obs
    buffer for stage-1 training. At inference, only the encoder's mu is used."""

    def __init__(self, obs_dim, latent_dim=8, hidden=128, device=None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = VAEEncoder(obs_dim, latent_dim, hidden).to(self.device)
        self.decoder = VAEDecoder(latent_dim, obs_dim, hidden).to(self.device)
        self.buffer = []

    def add(self, obs):
        if isinstance(obs, torch.Tensor):
            t = obs.detach().cpu().float().reshape(-1)
        else:
            t = torch.from_numpy(obs.astype("float32")).reshape(-1)
        self.buffer.append(t)

    def stack(self):
        return torch.stack(self.buffer, dim=0)  # (N, obs_dim) on CPU

    def train(self, epochs=100, batch_size=64, lr=1e-4, beta=1.0, verbose=True):
        N = len(self.buffer)
        if N < batch_size:
            raise ValueError(
                f"VAE-Embedding needs at least batch_size={batch_size} obs; got {N}."
            )

        # TensorDataset over the CPU-stacked buffer; DataLoader handles per-epoch
        # shuffling and batching. drop_last=True keeps the old N // batch_size
        # full-batch semantics (and the minimum-size guard above).
        loader = DataLoader(
            TensorDataset(self.stack()),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )

        pbar = tqdm(range(epochs), desc="VAE-Embedding train", disable=not verbose)
        for _ in pbar:
            self.encoder.train()
            self.decoder.train()
            recon_sum = 0.0
            kl_sum = 0.0

            for (x,) in loader:
                x = x.to(self.device)

                mu, logvar = self.encoder(x)
                z = reparameterize(mu, logvar)
                recon = self.decoder(z)
                loss, recon_term, kl_term = vae_loss(x, recon, mu, logvar, beta=beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                recon_sum += recon_term.item()
                kl_sum += kl_term.item()

            if verbose:
                num_batches = len(loader)
                pbar.set_postfix(
                    recon=f"{recon_sum / num_batches:.4f}",
                    kl=f"{kl_sum / num_batches:.4f}",
                )

    def encode(self, obs):
        """obs: (N, obs_dim) tensor -> (N, latent_dim) tensor. Uses the
        encoder mean (deterministic) for the latent state, matching the
        LUSR paper's framing: "the latent embedding of VAE [..] used as
        an internal latent state representation in RL"."""
        self.encoder.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            mu, _ = self.encoder(obs)
            return mu

    def save(self, path):
        # We only need the encoder at inference, but the decoder is small so
        # we save both for completeness (cheap and lets us inspect reconstructions).
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
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
        core.encoder.load_state_dict(ckpt["encoder"])
        core.decoder.load_state_dict(ckpt["decoder"])
        core.encoder.to(core.device)
        core.decoder.to(core.device)
        return core


@Registry.register_sim2real_model("vae_embedding")
class VAEEmbeddingModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.cores = [
            VAEEmbeddingCore(
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
            "beta": config.get("beta", 1.0),  # 1.0 = vanilla VAE (LUSR paper's VAE-Embedding)
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 100)

    def prepare_latent_train_episode(self, episode_id, trainer):
        # Multi-domain collection, matching the LUSR paper's shared dataset
        # across baselines.
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        # No domain tagging; vanilla VAE just learns a smooth latent over the
        # pooled multi-domain obs distribution.
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
        self.cores = [VAEEmbeddingCore.load(p, device=self.device) for p in paths]
