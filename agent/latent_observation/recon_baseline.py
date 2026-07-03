"""Reconstruction-baseline observation model.

Unlike every other method in this package (LUSR / DARLA / VAE-Embedding /
CURL / ATC), this baseline does NOT hand the policy a latent code. It trains a
plain bottleneck autoencoder on the pooled multi-domain stage-1 observations
and, at stage 2, the policy consumes the DECODER OUTPUT -- the reconstructed
observation, in the original observation space:

    obs -> encoder -> bottleneck(latent_dim) -> decoder -> obs_hat -> policy

So through the trainer's contract:
  * encode()      returns decoder(encoder(obs))  -- shape (N, obs_dim)
  * latent_dim()  returns obs_dim                -- Q-net sized to the raw obs

The idea: the autoencoder is trained only on observations from the (DR-sampled)
training domains, so its bottleneck learns the structure of plausible
observations. At eval time a degraded/real observation is projected through the
bottleneck back onto that manifold -- the reconstruction acts as a learned
"cleaner" for the observation, and the RL agent trains/acts on cleaned
observations rather than on an abstract latent.

`latent_dim` here is therefore the BOTTLENECK width (compression strength; the
implicit denoising knob), not the size of what the agent sees -- the agent
always sees obs_dim.

Kept deliberately minimal (deterministic AE, per-element MSE) so it is the
pure reconstruction baseline: no KL/variational machinery (VAE-Embedding), no
disentanglement pressure (DARLA), no cycle-consistency (LUSR), no contrastive
objective (CURL/ATC). Stage-1 collection matches all other baselines
(episode 0 = default sim config, later episodes = DR-sampled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common.registry import Registry
from .base import LatentObservationModel


class AEEncoder(nn.Module):
    """MLP encoder obs -> bottleneck code."""

    def __init__(self, obs_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class AEDecoder(nn.Module):
    """MLP decoder bottleneck code -> reconstructed obs."""

    def __init__(self, latent_dim, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class ReconBaselineCore:
    """Single-intersection autoencoder. Holds encoder + decoder + a flat obs
    buffer for stage-1 training. At inference BOTH halves are used: the agent
    consumes the reconstruction, not the code."""

    def __init__(self, obs_dim, latent_dim=8, hidden=128, device=None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = AEEncoder(obs_dim, latent_dim, hidden).to(self.device)
        self.decoder = AEDecoder(latent_dim, obs_dim, hidden).to(self.device)
        self.buffer = []

    def add(self, obs):
        if isinstance(obs, torch.Tensor):
            t = obs.detach().cpu().float().reshape(-1)
        else:
            t = torch.from_numpy(obs.astype("float32")).reshape(-1)
        self.buffer.append(t)

    def stack(self):
        return torch.stack(self.buffer, dim=0)  # (N, obs_dim) on CPU

    def train(self, epochs=100, batch_size=64, lr=1e-4, verbose=True):
        N = len(self.buffer)
        if N < batch_size:
            raise ValueError(
                f"Reconstruction baseline needs at least batch_size={batch_size} obs; got {N}."
            )

        loader = DataLoader(
            TensorDataset(self.stack()),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )

        pbar = tqdm(range(epochs), desc="Recon-baseline train", disable=not verbose)
        for _ in pbar:
            self.encoder.train()
            self.decoder.train()
            loss_sum = 0.0

            for (x,) in loader:
                x = x.to(self.device)
                recon = self.decoder(self.encoder(x))
                loss = F.mse_loss(recon, x, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            if verbose:
                pbar.set_postfix(mse=f"{loss_sum / len(loader):.4f}")

    def encode(self, obs):
        """obs: (N, obs_dim) tensor -> (N, obs_dim) tensor: the RECONSTRUCTED
        observation. This is what the policy consumes -- observation space,
        not the bottleneck code."""
        self.encoder.eval()
        self.decoder.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            return self.decoder(self.encoder(obs))

    def save(self, path):
        # Both halves are required at inference (the agent consumes the
        # reconstruction), unlike the latent methods where the decoder is
        # training-only state.
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


@Registry.register_sim2real_model("recon_baseline")
class ReconBaselineModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.cores = [
            ReconBaselineCore(
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
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 100)

    def prepare_latent_train_episode(self, episode_id, trainer):
        # Multi-domain collection, matching all other baselines (episode 0 =
        # source/default domain, later episodes = freshly DR-sampled domains).
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        # No domain tagging; the AE learns the pooled multi-domain obs manifold.
        self.cores[intersection_id].add(obs)

    def train(self):
        for core in self.cores:
            core.train(**self._training_kwargs)

    def latent_dim(self, intersection_id):
        # The policy consumes the RECONSTRUCTION, so its input dim is the raw
        # observation dim -- not the bottleneck width.
        return self.cores[intersection_id].obs_dim

    def encode(self, intersection_id, obs_tensor):
        return self.cores[intersection_id].encode(obs_tensor)

    def save(self, paths):
        for core, path in zip(self.cores, paths):
            core.save(path)

    def load(self, paths):
        self.cores = [ReconBaselineCore.load(p, device=self.device) for p in paths]
