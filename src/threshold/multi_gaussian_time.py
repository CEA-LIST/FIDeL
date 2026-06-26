import torch
import os
from scipy.stats import norm


class Multi_gaussian_time:
    def __init__(self, cfg, scores, alpha: float = 0.95):
        self.cfg = cfg
        self.alpha = alpha
        self.alpha_all = cfg.threshold.alpha_all
        self.N = cfg.threshold.N
        self.N_all = cfg.threshold.N_all

        self.param_list = [({"alpha": alpha, "N": self.N}) for alpha in self.alpha_all]
        self.param_list.extend([{"alpha": self.alpha, "N": N} for N in self.N_all])

        self.all_mean_per_timestep, self.all_std_per_timestep = (
            self.compute_expert_gaussians_all(scores)
        )
        self.mean_per_timestep, self.std_per_timestep = self.compute_expert_gaussians(
            scores
        )

        # Threshold calculation based on the absolute value of the normalized deviation (z-score)
        self.threshold_all_N = self._compute_threshold_all_N()
        self.threshold_all_alpha = self._compute_threshold_all_alpha()

    def _compute_threshold_all_alpha(self):
        """
        Converts confidence level (alpha) to z-score threshold
        """
        # Alpha -> z-score threshold (e.g. 1.96 for 95%)
        alpha_tensor = torch.tensor(self.alpha_all, dtype=torch.float32)
        z = torch.tensor(
            norm.ppf((1 + alpha_tensor.numpy()) / 2), dtype=torch.float32
        ).unsqueeze(1)  # shape: [nb_alpha, 1]
        threshold_values = self.mean_per_timestep + z * self.std_per_timestep
        return threshold_values

    def _compute_threshold_all_N(self):
        """
        Converts confidence level (alpha) to z-score threshold
        """
        # Alpha -> z-score threshold (e.g. 1.96 for 95%)
        z = norm.ppf((1 + self.alpha) / 2)  # bilateral threshold
        mean_tensor = torch.stack(
            self.all_mean_per_timestep
        )  # shape: [nb_N, nb_timesteps]
        std_tensor = torch.stack(
            self.all_std_per_timestep
        )  # shape: [nb_N, nb_timesteps]

        threshold_tensor = mean_tensor + z * std_tensor  # shape: [nb_N, nb_timesteps]
        return threshold_tensor

    def compute_expert_gaussians_all(self, scores):
        """
        Computes the mean and standard deviation of local means for each timestep
        of each episode, without padding, using only available neighbors.

        Args:
            scores (torch.Tensor): Tensor of shape [nb_episodes, nb_timesteps]
            N (list of int): Number of neighbors to the left and right

        Returns:
            list of Tuple[torch.Tensor, torch.Tensor]: means and standard deviations, shape [nb_timesteps], for N_all
        """
        nb_episodes, nb_timesteps = scores.shape
        all_mean = []
        all_std = []
        for N in self.N_all:
            # Prepare a tensor to store local means
            local_means = torch.empty_like(scores)

            for i in range(nb_episodes):
                for j in range(nb_timesteps):
                    start = max(0, j - N)
                    end = min(nb_timesteps, j + N + 1)
                    local_means[i, j] = scores[
                        i, start:end
                    ].mean()  # mean of 2N scores for each timestep

            # Mean and standard deviation across episodes
            all_mean.append(local_means.mean(dim=0))  # [nb_timesteps]
            all_std.append(local_means.std(dim=0, unbiased=False))  # [nb_timesteps]

        return all_mean, all_std  # [nb_N, nb_timesteps]

    def compute_expert_gaussians(self, scores):
        nb_episodes, nb_timesteps = scores.shape

        # Prepare a tensor to store local means
        local_means = torch.empty_like(scores)

        for i in range(nb_episodes):
            for j in range(nb_timesteps):
                start = max(0, j - self.N)
                end = min(nb_timesteps, j + self.N + 1)
                local_means[i, j] = scores[
                    i, start:end
                ].mean()  # mean of 2N scores for each timestep

        # Mean and standard deviation across episodes
        mean_per_timestep = local_means.mean(dim=0)  # [nb_timesteps]
        std_per_timestep = local_means.std(dim=0, unbiased=False)  # [nb_timesteps]

        return mean_per_timestep, std_per_timestep

    def __call__(self, score: torch.Tensor, heatmap: torch.Tensor, data: dict) -> bool:
        """
        Returns True if the score is anomalous (above the threshold)
        """
        if isinstance(score, torch.Tensor):
            score = score.item()

        index = data["frame_index"].item()

        score_tensor = torch.tensor(score)

        # Vectorized comparison on alpha thresholds (shape: [nb_alpha])
        is_anomaly_alpha = (
            score_tensor > self.threshold_all_alpha[:, index]
        )  # [nb_alpha]

        # Vectorized comparison on N thresholds (shape: [nb_N])
        is_anomaly_N = score_tensor > self.threshold_all_N[:, index]  # [nb_N]

        # Concatenates all results into a single boolean vector [nb_alpha + nb_N]
        is_anomaly_all = torch.cat(
            [is_anomaly_alpha, is_anomaly_N], dim=0
        )  # shape: [nb_alpha + nb_N]

        return is_anomaly_all, self.threshold_all_N[:, index]
