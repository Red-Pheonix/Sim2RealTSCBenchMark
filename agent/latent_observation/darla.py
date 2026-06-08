"""DARLA latent-observation model (Higgins et al. 2017, arXiv:1707.08475).

Three stages: (0) train a denoising autoencoder, freeze it; (1) train a
beta-VAE whose reconstruction loss is measured in the frozen DAE feature space
J(.) (Eq. 2); (2) freeze the encoder and feed its mean mu to the RL agent.
Stage 2 is handled by the shared trainer; this file owns stages 0-1 and encode.

See DARLA_README.md for a step-by-step walkthrough and the list of deliberate
deviations from the paper.
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


def mask_noise(x, mask_prob):
    """Masking noise: zero each feature independently w.p. mask_prob (the
    vector analogue of the paper's rectangular pixel occlusion, A.3.1)."""
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


class DAE(nn.Module):
    """Denoising autoencoder. Trained in stage 0, then frozen; `features(x)` is
    the perceptual space J(.) the beta-VAE reconstructs into. J is read off the
    last decoder hidden layer pre-activation (paper A.3.2, footnote 8)."""

    def __init__(self, obs_dim, hidden=128, feature_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim), nn.ReLU(),
        )
        self.dec_hidden = nn.Linear(feature_dim, hidden)
        self.dec_act = nn.ReLU()
        self.dec_out = nn.Linear(hidden, obs_dim)

    def features(self, x):
        return self.dec_hidden(self.enc(x))

    def forward(self, x):
        return self.dec_out(self.dec_act(self.features(x)))


class VAEEncoder(nn.Module):
    """obs -> (mu, logvar) of the latent Gaussian."""

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
    """Gaussian decoder z -> p(x|z), i.e. per-dim (mu, logvar) (paper A.3.2)."""

    def __init__(self, latent_dim, obs_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, obs_dim)
        self.fc_logvar = nn.Linear(hidden, obs_dim)

    def forward(self, z):
        h = self.trunk(z)
        return self.fc_mu(h), self.fc_logvar(h)

    def rsample(self, z):
        # full sample x_hat ~ p(x|z), not the mean (paper footnote 7)
        mu, logvar = self.forward(z)
        return reparameterize(mu, logvar)


def darla_loss(x, recon_x, mu, logvar, dae, beta):
    """beta-VAE_DAE objective (Eq. 2/3): perceptual L2 in the frozen DAE space
    plus beta * KL. Recon is summed over features, KL over latents, both meaned
    over the batch -- the paper's scaling, so beta matches its reported values.
    J(x) is detached (fixed target); gradients flow only through J(x_hat)."""
    feat_recon = dae.features(recon_x)
    feat_target = dae.features(x).detach()
    recon = ((feat_recon - feat_target) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon + beta * kl, recon, kl


class DARLACore:
    """One DAE + beta-VAE per intersection, plus a flat stage-1 obs buffer.
    At inference only the encoder mean mu is used as the RL state."""

    def __init__(self, obs_dim, latent_dim=8, hidden=128, dae_hidden=128,
                 dae_feature_dim=64, device=None):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.dae_hidden = dae_hidden
        self.dae_feature_dim = dae_feature_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dae = DAE(obs_dim, dae_hidden, dae_feature_dim).to(self.device)
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

    def _train_dae(self, loader, epochs, lr, mask_prob, verbose):
        optimizer = torch.optim.Adam(self.dae.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc="DARLA DAE train", disable=not verbose)
        for _ in pbar:
            self.dae.train()
            loss_sum = 0.0
            for (x,) in loader:
                x = x.to(self.device)
                recon = self.dae(mask_noise(x, mask_prob))
                loss = F.mse_loss(recon, x, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            if verbose:
                pbar.set_postfix(dae_recon=f"{loss_sum / len(loader):.4f}")

        # freeze: the DAE is now a fixed perceptual feature space
        self.dae.eval()
        for p in self.dae.parameters():
            p.requires_grad = False

    def train(self, epochs=300, batch_size=64, lr=1e-4, beta=10.0,
              dae_epochs=100, dae_lr=1e-3, dae_mask_prob=0.2, verbose=True):
        N = len(self.buffer)
        if N < batch_size:
            raise ValueError(
                f"DARLA needs at least batch_size={batch_size} obs; got {N}."
            )

        loader = DataLoader(
            TensorDataset(self.stack()),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # stage 0: DAE (then frozen); stage 1: beta-VAE against J(.)
        self._train_dae(loader, dae_epochs, dae_lr, dae_mask_prob, verbose)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        pbar = tqdm(range(epochs), desc="DARLA beta-VAE train", disable=not verbose)
        for _ in pbar:
            self.encoder.train()
            self.decoder.train()
            recon_sum = 0.0
            kl_sum = 0.0
            for (x,) in loader:
                x = x.to(self.device)
                mu, logvar = self.encoder(x)
                z = reparameterize(mu, logvar)
                x_hat = self.decoder.rsample(z)
                loss, recon_term, kl_term = darla_loss(
                    x, x_hat, mu, logvar, self.dae, beta
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                recon_sum += recon_term.item()
                kl_sum += kl_term.item()
            if verbose:
                nb = len(loader)
                pbar.set_postfix(
                    recon=f"{recon_sum / nb:.4f}", kl=f"{kl_sum / nb:.4f}"
                )

    def encode(self, obs):
        # deterministic mu = the latent the policy consumes (stage 2)
        self.encoder.eval()
        obs = obs.to(self.device).float()
        with torch.no_grad():
            mu, _ = self.encoder(obs)
            return mu

    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "dae": self.dae.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "latent_dim": self.latent_dim,
                    "hidden": self.hidden,
                    "dae_hidden": self.dae_hidden,
                    "dae_feature_dim": self.dae_feature_dim,
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
            dae_hidden=cfg["dae_hidden"],
            dae_feature_dim=cfg["dae_feature_dim"],
            device=device,
        )
        core.encoder.load_state_dict(ckpt["encoder"])
        core.decoder.load_state_dict(ckpt["decoder"])
        core.dae.load_state_dict(ckpt["dae"])
        for m in (core.encoder, core.decoder, core.dae):
            m.to(core.device)
        return core


@Registry.register_sim2real_model("darla")
class DARLAModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.cores = [
            DARLACore(
                obs_dim=obs_dims[i],
                latent_dim=config.get("latent_dim", 8),
                hidden=config.get("hidden", 128),
                dae_hidden=config.get("dae_hidden", 128),
                dae_feature_dim=config.get("dae_feature_dim", 64),
                device=device,
            )
            for i in range(num_intersections)
        ]
        self._training_kwargs = {
            "epochs": config.get("epochs", 300),
            "batch_size": config.get("batch_size", 64),
            "lr": config.get("lr", 1e-4),
            "beta": config.get("beta", 10.0),
            # DAE uses a higher lr than the beta-VAE by design (paper A.3.1/A.3.2)
            "dae_epochs": config.get("dae_epochs", 100),
            "dae_lr": config.get("dae_lr", 1e-3),
            "dae_mask_prob": config.get("dae_mask_prob", 0.2),
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 100)

    def prepare_latent_train_episode(self, episode_id, trainer):
        # episode 0 = source domain, later episodes = fresh DR domains, so the
        # pooled stage-1 buffer spans the range of generative factors
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        self.cores[intersection_id].add(obs)  # flat buffer, no domain tags

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
        self.cores = [DARLACore.load(p, device=self.device) for p in paths]
