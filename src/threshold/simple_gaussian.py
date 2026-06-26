import torch
from scipy.stats import norm


class Simple_gaussian:
    def __init__(self, cfg, mean: torch.Tensor, std: torch.Tensor):
        """
        mean: global mean of expert scores (scalar Tensor)
        std: global standard deviation of expert scores (scalar Tensor)
        alpha: confidence level (e.g. 0.60, 0.95, 0.99)
        """
        assert mean.ndim == 0 and std.ndim == 0, (
            "mean and std must be scalars (0D Tensors)"
        )
        self.cfg = cfg
        self.mean = mean.item()
        self.std = std.item()

        self.alpha_all = cfg.threshold.alpha_all
        self.param_list = [({"alpha": alpha}) for alpha in self.alpha_all]

        # Threshold calculation based on the absolute value of the normalized deviation (z-score)
        self.threshold_all = [
            self._compute_threshold(alpha) for alpha in self.alpha_all
        ]

    def _compute_threshold(self, alpha):
        """
        Converts confidence level (alpha) to z-score threshold
        """
        # Alpha -> z-score threshold (e.g. 1.96 for 95%)
        z = norm.ppf((1 + alpha) / 2)  # bilateral threshold
        threshold_value = self.mean + z * self.std
        return threshold_value

    def __call__(self, score: torch.Tensor, heatmap: torch.Tensor, data: dict) -> bool:
        """
        Returns True if the score is anomalous (above the threshold)
        """
        if isinstance(score, torch.Tensor):
            score = score.item()

        return [score > thr for thr in self.threshold_all]
