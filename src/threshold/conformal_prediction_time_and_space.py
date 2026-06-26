import torch
import numpy as np
import random


class ConformalPredictionTimeAndSpace:
    def __init__(self, cfg, scores: torch.Tensor, heatmaps: torch.Tensor):
        """
        Implements temporal AND spatial Conformal Prediction in a single class

        Args:
            cfg: config containing N (used if scalA(t) is local)
            scores: [nb_episodes, T]
            alpha: rejection level (e.g. 0.05 for 95% confidence)
            scal_variant: "variant1" or "variant2"
        """

        self.cfg = cfg
        self.alpha_time = cfg.threshold.alpha_time
        self.alpha_space = cfg.threshold.alpha_space
        self.split_rate = cfg.threshold.split_rate
        self.scal_variant_time = cfg.threshold.scal_variant_time
        assert self.scal_variant_time in ["variant1", "variant2"], (
            "scal_variant must be 'variant1' or 'variant2'"
        )
        self.scal_variant_space = cfg.threshold.scal_variant_space
        assert self.scal_variant_space in ["variant1", "variant2"], (
            "scal_variant must be 'variant1' or 'variant2'"
        )
        self.patch_percentage = cfg.threshold.percentage_patch_detection
        self.nb_episodes, self.T, self.nb_patch = heatmaps.shape

        # Step 1: Split DcalA_time / DcalB_time - DcalA_space / DcalB_space
        self.split_datasets(scores, heatmaps)

        # Step 2:
        # Mean μₜ
        self.mu_t = self.DcalA_time.mean(dim=0)
        # Mean µp
        self.mu_p = self.DcalA_space.mean(dim=(0, 1))

        # Step 3: scalA(t), scalA(p)
        # time
        if self.scal_variant_time == "variant1":
            self.scalA_t = self.compute_scalA_variant1_time()
        else:
            self.scalA_t = self.compute_scalA_variant2_time()
        # space
        if self.scal_variant_space == "variant1":
            self.scalA_p = self.compute_scalA_variant1_space()
        else:
            self.scalA_p = self.compute_scalA_variant2_space()

        # Step 4: compute h_time, h_space and thresholds ηₜ and ηp
        self.h_time = self.compute_quantile_h_time()
        self.h_space = self.compute_quantile_h_space()
        self.threshold_time = self.mu_t + self.h_time * self.scalA_t
        self.threshold_space = self.mu_p + self.h_space * self.scalA_p

    def split_datasets(self, scores, heatmaps):
        all_indices = list(range(self.nb_episodes))
        random.shuffle(all_indices)
        N1 = int(self.split_rate * self.nb_episodes)
        A_idx = all_indices[:N1]
        B_idx = all_indices[N1:]
        self.DcalA_time = scores[A_idx, :]  # [N1, T]
        self.DcalB_time = scores[B_idx, :]  # [N2, T]
        self.DcalA_space = heatmaps[A_idx, :]  # [N1, T, nb_patch]
        self.DcalB_space = heatmaps[B_idx, :]  # [N2, T, nb_patch]

    def compute_scalA_variant1_time(self):
        """
        scalA(t) = 1 / T (constant for all t)
        """
        return torch.ones(self.T) / self.T

    def compute_scalA_variant2_time(self):
        """
        scalA(t) = max_k |DMₖₜ - μₜ| on DcalA
        """
        abs_dev = torch.abs(self.DcalA_time - self.mu_t)  # [N1, T]
        return abs_dev.max(dim=0).values  # [T]

    def compute_scalA_variant1_space(self):
        """
        scalA(p) = 1 / nb_patch (constant for all p)
        """
        return torch.ones(self.nb_patch) / self.nb_patch

    def compute_scalA_variant2_space(self):
        """
        scalA(p) = max_k |DMₖp - μp| on DcalA
        """
        abs_dev = torch.abs(self.DcalA_space - self.mu_p)  # [N1, T, nb_patch]
        abs_dev = abs_dev.max(dim=0).values  # [T, nb_patch]
        return abs_dev.max(dim=0).values  # [nb_patch]

    def compute_quantile_h_space(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB_space.shape[0]
        deviations = []

        for j in range(N2):
            for m in range(self.T):
                epsilon = 1e-6
                self.scalA_p[self.scalA_p == 0] = epsilon
                dev_j_t = (
                    self.mu_p - self.DcalB_space[j, m]
                ) / self.scalA_p  # [nb_patch]
                Dj = dev_j_t.max().item()  # max on patches
                deviations.append(Dj)

        h = np.quantile(deviations, 1 - self.alpha_space)
        return h

    def compute_quantile_h_time(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB_time.shape[0]
        deviations = []

        for j in range(N2):
            epsilon = 1e-6
            self.scalA_t[self.scalA_t == 0] = epsilon
            dev_j = (self.mu_t - self.DcalB_time[j]) / self.scalA_t  # [T]
            Dj = dev_j.max().item()  # max on time
            deviations.append(Dj)

        h = np.quantile(deviations, 1 - self.alpha_time)
        return h

    def has_at_least_X_percent_true(self, tensor):
        total = tensor.numel()  # total number of elements
        n_true = tensor.sum().item()  # number of True (since they equal 1)
        return n_true / total >= self.patch_percentage

    def __call__(
        self, score: torch.Tensor, heatmap: torch.Tensor, data: dict = None
    ) -> bool:
        # time
        t = data["frame_index"].item()
        score = score.item() if isinstance(score, torch.Tensor) else score
        threshold_time = self.threshold_time[t]
        detect_time = score > self.threshold_time[t]

        # space
        anomaly_flag = heatmap >= self.threshold_space
        detect_space = self.has_at_least_X_percent_true(
            anomaly_flag
        )  # if 10% of patches detect an anomaly, flag is set to true

        assert self.cfg.threshold.time_AND_space ^ self.cfg.threshold.time_OR_space, (
            "choose between OR & AND, can't be both or neither"
        )
        if self.cfg.threshold.time_AND_space:
            return detect_time and detect_space  # we detect if either detects or both
        elif self.cfg.threshold.time_OR_space:
            return detect_time or detect_space  # we only detect if both detects
