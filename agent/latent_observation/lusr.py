"""LUSR latent-observation model.

Stage 1: collect observations across N DR-sampled domains (episode 0 uses the
default sim config; later episodes sample a fresh DR config). Train one
cycle-consistent disentangled VAE per intersection on the collected per-domain
data.

Stage 2: a fresh PressLight per intersection consumes the frozen content
latent through the trainer's get_ob monkey-patch.

Model code (Encoder / Decoder / DisentangledVAE / LUSRDataset / LUSR) is
lifted from the test.py sandbox; see notes/lusr.md for the deeper writeup and
references to the original paper / repo (github.com/KarlXing/LUSR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.registry import Registry
from .base import LatentObservationModel


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

    def save(self, path):
        torch.save(
            {
                "vae": self.vae.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "content_dim": self.content_dim,
                    "class_dim": self.class_dim,
                    "hidden": self.hidden,
                },
            },
            path,
        )

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
        return lusr


@Registry.register_sim2real_model("lusr")
class LUSRModel(LatentObservationModel):
    def __init__(self, config, num_intersections, obs_dims, device):
        super().__init__(config, num_intersections, obs_dims, device)
        self.lusrs = [
            LUSR(
                obs_dim=obs_dims[i],
                content_dim=config.get("content_dim", 8),
                class_dim=config.get("class_dim", 4),
                hidden=config.get("hidden", 128),
                device=device,
            )
            for i in range(num_intersections)
        ]
        self._training_kwargs = {
            "epochs": config.get("epochs", 50),
            "batch_size": config.get("batch_size", 64),
            "beta": config.get("beta", 10.0),
            "bloss_coef": config.get("bloss_coef", 1.0),
            "lr": config.get("lr", 1e-4),
        }

    @property
    def n_pretrain_episodes(self):
        return self.config.get("pretrain_episodes", 50)

    def prepare_latent_train_episode(self, episode_id, trainer):
        # Episode 0 collects from the default sim config (source domain); later
        # episodes sample a fresh DR config to provide multi-domain diversity
        # for the disentangled VAE's class code.
        if episode_id == 0:
            trainer.apply_default_sim_domain()
        else:
            trainer.apply_new_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        self.lusrs[intersection_id].save_batch(
            torch.from_numpy(obs.astype("float32")), domain_id=episode_id
        )

    def train(self):
        for i, lusr in enumerate(self.lusrs):
            lusr.train(**self._training_kwargs)

    def latent_dim(self, intersection_id):
        return self.lusrs[intersection_id].content_dim

    def encode(self, intersection_id, obs_tensor):
        return self.lusrs[intersection_id].encode(obs_tensor)

    def save(self, paths):
        for lusr, path in zip(self.lusrs, paths):
            lusr.save(path)

    def load(self, paths):
        self.lusrs = [LUSR.load(p, device=self.device) for p in paths]
