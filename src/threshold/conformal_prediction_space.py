import torch
import numpy as np
import random


class ConformalPredictionSpace:
    def __init__(self, cfg, scores: torch.Tensor, heatmaps):
        """
        Implements spatial Conformal Prediction with choice of scalA(t)
        !!! /!\ Only works if the encoder is DinoV2 which outputs features per patch !!!

        Args:
            cfg: config containing N (used if scalA(t) is local)
            scores: [nb_episodes, T]
            heatmaps: [nb_episodes, T, nb_patch]
        """

        self.cfg = cfg
        self.alpha = cfg.threshold.alpha
        self.split_rate = cfg.threshold.split_rate
        self.patch_percentage = cfg.threshold.percentage_patch_detection
        self.scal_variant = cfg.threshold.scal_variant
        assert self.scal_variant in ["variant1", "variant2"], (
            "scal_variant must be 'variant1' or 'variant2'"
        )
        self.nb_episodes, self.T, self.nb_patch = heatmaps.shape

        # Step 1: Split DcalA / DcalB
        self.split_datasets(heatmaps)

        # Step 2: Mean μp on patches
        self.mu_p = self.DcalA.mean(dim=(0, 1))

        self.device = heatmaps.device
        self.mu_p = self.mu_p.to(self.device)

        # Step 3: scalA(t)
        if self.scal_variant == "variant1":
            self.scalA_p = self.compute_scalA_variant1()
        else:
            self.scalA_p = self.compute_scalA_variant2()

        # Step 4: compute h and final threshold ηp
        self.h = self.compute_quantile_h()
        self.threshold = self.mu_p + self.h * self.scalA_p

    def split_datasets(self, heatmaps):
        all_indices = list(range(self.nb_episodes))
        random.shuffle(all_indices)
        N1 = int(self.split_rate * self.nb_episodes)
        A_idx = all_indices[:N1]
        B_idx = all_indices[N1:]
        self.DcalA = heatmaps[A_idx, :]  # [N1, T, nb_patch]
        self.DcalB = heatmaps[B_idx, :]  # [N2, T, nb_patch]

    def compute_scalA_variant1(self):
        """
        scalA(p) = 1 / nb_patch (constant for all p)
        """
        return torch.ones(self.nb_patch, device=self.device) / self.nb_patch

    def compute_scalA_variant2(self):
        """
        scalA(p) = max_k |DMₖp - μp| on DcalA
        """
        abs_dev = torch.abs(self.DcalA - self.mu_p)  # [N1, T, nb_patch]
        abs_dev = abs_dev.max(dim=0).values  # [T, nb_patch]
        return abs_dev.max(dim=0).values.to(self.device)  # [nb_patch]

    def compute_quantile_h(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB.shape[0]
        deviations = []

        for j in range(N2):
            for m in range(self.T):
                epsilon = 1e-6
                self.scalA_p[self.scalA_p == 0] = epsilon
                dev_j_t = (self.mu_p - self.DcalB[j, m]) / self.scalA_p  # [nb_patch]
                Dj = dev_j_t.max().item()  # max on patches
                deviations.append(Dj)

        h = np.quantile(deviations, 1 - self.alpha)
        return h

    def has_at_least_X_percent_true(self, tensor):
        total = tensor.numel()  # total number of elements
        n_true = tensor.sum().item()  # number of True (since they equal 1)
        return n_true / total >= self.patch_percentage

    def __call__(
        self, score: torch.Tensor, heatmap: torch.Tensor, data: dict = None
    ) -> bool:
        # t = data["frame_index"].item()
        # score = score.item() if isinstance(score, torch.Tensor) else score
        anomaly_flag = heatmap >= self.threshold
        if self.has_at_least_X_percent_true(
            anomaly_flag
        ):  # if 10% of patches detect an anomaly, trigger
            return True
        else:
            return False
