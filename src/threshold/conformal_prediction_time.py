import torch
import numpy as np
import random


class ConformalPredictionTime:
    def __init__(self, cfg, scores: torch.Tensor):
        """
        Implements temporal Conformal Prediction with choice of scalA(t)

        Args:
            cfg: config containing N (used if scalA(t) is local)
            scores: [nb_episodes, T]
            alpha: rejection level (e.g. 0.05 for 95% confidence)
            scal_variant: "variant1" or "variant2"
        """

        self.cfg = cfg
        self.alpha = cfg.threshold.alpha
        self.split_rate = cfg.threshold.split_rate
        self.scal_variant = cfg.threshold.scal_variant
        assert self.scal_variant in ["variant1", "variant2"], (
            "scal_variant must be 'variant1' or 'variant2'"
        )
        self.nb_episodes, self.T = scores.shape

        # Step 1: Split DcalA / DcalB
        self.split_datasets(scores)

        # Step 2: Mean μₜ
        self.mu_t = self.DcalA.mean(dim=0)

        # Step 3: scalA(t)
        if self.scal_variant == "variant1":
            self.scalA_t = self.compute_scalA_variant1()
        else:
            self.scalA_t = self.compute_scalA_variant2()

        # Step 4: compute h and final threshold ηₜ
        self.h = self.compute_quantile_h()
        self.threshold = self.mu_t + self.h * self.scalA_t

    def split_datasets(self, scores):
        all_indices = list(range(self.nb_episodes))
        random.shuffle(all_indices)
        N1 = int(self.split_rate * self.nb_episodes)
        A_idx = all_indices[:N1]
        B_idx = all_indices[N1:]
        self.DcalA = scores[A_idx, :]  # [N1, T]
        self.DcalB = scores[B_idx, :]  # [N2, T]

    def compute_scalA_variant1(self):
        """
        scalA(t) = 1 / T (constant for all t)
        """
        return torch.ones(self.T) / self.T

    def compute_scalA_variant2(self):
        """
        scalA(t) = max_k |DMₖₜ - μₜ| on DcalA
        """
        abs_dev = torch.abs(self.DcalA - self.mu_t)  # [N1, T]
        return abs_dev.max(dim=0).values  # [T]

    def compute_quantile_h(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB.shape[0]
        deviations = []

        for j in range(N2):
            epsilon = 1e-6
            self.scalA_t[self.scalA_t == 0] = epsilon
            dev_j = (self.mu_t - self.DcalB[j]) / self.scalA_t  # [T]
            Dj = dev_j.max().item()  # max on time
            deviations.append(Dj)

        h = np.quantile(deviations, 1 - self.alpha)
        return h

    def __call__(self, score: torch.Tensor, heatmap: torch.Tensor, data: dict) -> bool:
        t = data["frame_index"].item()
        score = score.item() if isinstance(score, torch.Tensor) else score
        threshold = self.threshold[t]
        return score > self.threshold[t]
