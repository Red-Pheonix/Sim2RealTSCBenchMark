"""Base contract for latent-observation encoder models.

A LatentObservationModel is a passive data-sink + encoder. The trainer
(LatentObservationTrainer) drives stage-1 rollouts and stage-2 RL; the model
owns the encoder weights and the per-method policy decisions (e.g. when to use
DR vs default sim config during stage 1).

Subclasses register themselves under `sim2real_model_mapping` and override the
NotImplementedError methods below.
"""

import os


class LatentObservationModel:
    def __init__(self, config, num_intersections, obs_dims, device):
        self.config = config
        self.num_intersections = num_intersections
        self.obs_dims = obs_dims
        self.device = device

    @property
    def n_pretrain_episodes(self) -> int:
        raise NotImplementedError

    @property
    def force_retrain(self) -> bool:
        return self.config.get("force_retrain", False)

    def prepare_latent_train_episode(self, episode_id, trainer):
        """Called once at the start of each stage-1 collection episode, before
        the trainer resets the env. The model uses this hook to choose the
        sim's observation config for the upcoming rollout -- typically by
        calling trainer.apply_default_sim_domain() or
        trainer.apply_new_sim_domain(). The trainer reference is passed at call
        time only; the model should not store it.

        Default: stay on the default sim config every episode."""
        trainer.apply_default_sim_domain()

    def save_batch(self, intersection_id, obs, episode_id):
        """Record one (1, obs_dim) numpy observation for intersection_id from
        stage-1 episode episode_id."""
        raise NotImplementedError

    def train(self):
        """Train the encoder(s) on the data collected via save_batch."""
        raise NotImplementedError

    def latent_dim(self, intersection_id) -> int:
        """Output dim of the encoder for intersection_id. The trainer sizes
        each PressLight Q-net to (latent_dim + phase_extra)."""
        raise NotImplementedError

    def encode(self, intersection_id, obs_tensor):
        """obs_tensor: (N, obs_dim) torch.Tensor -> (N, latent_dim) torch.Tensor.
        Called from inside the agent's monkey-patched get_ob each env step."""
        raise NotImplementedError

    def cache_paths(self, base_dir):
        """Files the model expects to save to / load from. Default: one file
        per intersection under base_dir."""
        return [
            os.path.join(base_dir, f"intersection_{i}.pt")
            for i in range(self.num_intersections)
        ]

    def save(self, paths):
        raise NotImplementedError

    def load(self, paths):
        raise NotImplementedError
