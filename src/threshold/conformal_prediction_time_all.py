import torch
import numpy as np
import random


class ConformalPredictionTimeAll:
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
        self.alpha_all = cfg.threshold.alpha_all
        self.alpha_default = cfg.threshold.alpha_default
        self.split_rate = cfg.threshold.split_rate
        
        self.nb_episodes, self.T = scores.shape

        # Step 1: Split DcalA / DcalB
        self.split_datasets(scores)

        # Step 2: Mean μₜ
        self.mu_t = self.DcalA.mean(dim=0)

        # Step 3: scalA(t)
        self.scalA_t_var1 = self.compute_scalA_variant1()
        self.scalA_t_var2 = self.compute_scalA_variant2()

        # Step 4: compute h and final threshold ηₜ
        self.h_all_alphas_scalvar1 = self.compute_quantile_h_scalvar1()
        self.h_all_alphas_scalvar2 = self.compute_quantile_h_scalvar2()

        self.thresholds = []
        self.param_list = []

        # Variant 1
        for i, h in enumerate(self.h_all_alphas_scalvar1):
            threshold = self.mu_t + h * self.scalA_t_var1
            self.thresholds.append(threshold)
            self.param_list.append({
                "alpha": self.alpha_all[i],
                "split_rate": self.split_rate,
                "scal_variant": "variant1"
            })

        # Variant 2
        for i, h in enumerate(self.h_all_alphas_scalvar2):
            threshold = self.mu_t + h * self.scalA_t_var2
            self.thresholds.append(threshold)
            self.param_list.append({
                "alpha": self.alpha_all[i],
                "split_rate": self.split_rate,
                "scal_variant": "variant2"
            })


    def split_datasets(self, scores):
        all_indices = list(range(self.nb_episodes))
        random.shuffle(all_indices)
        
        if self.nb_episodes <= 1:
            # Fallback for dummy datasets with only 1 episode
            self.DcalA = scores
            self.DcalB = scores
            return
            
        N1 = int(self.split_rate * self.nb_episodes)
        # Ensure at least 1 item in each split if we have at least 2 episodes
        if N1 == 0:
            N1 = 1
        elif N1 == self.nb_episodes:
            N1 = self.nb_episodes - 1
            
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

    def compute_quantile_h_scalvar1(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB.shape[0]
        epsilon = 1e-6
        self.scalA_t_var1[self.scalA_t_var1 == 0] = epsilon

        deviations = [(torch.abs(self.mu_t - self.DcalB[j]) / self.scalA_t_var1).max().item() for j in range(N2)]
        h_all_alphas = [np.quantile(deviations, 1 - alpha) for alpha in self.alpha_all]
        
        print("\n[DEBUG] Conformal Prediction Var1 Stats:")
        print(f"  mu_t mean: {self.mu_t.mean().item():.4f}, mu_t max: {self.mu_t.max().item():.4f}")
        print(f"  Max deviation observed in DcalB: {max(deviations):.4f}")
        print(f"  h values (quantiles): {h_all_alphas}")
        
        return h_all_alphas
    
    def compute_quantile_h_scalvar2(self):
        """
        Computes h as the (1 - alpha) quantile of the normalized max_deviation on DcalB
        """
        N2 = self.DcalB.shape[0]
        epsilon = 1e-6
        self.scalA_t_var2[self.scalA_t_var2 == 0] = epsilon
        
        deviations = [(torch.abs(self.mu_t - self.DcalB[j]) / self.scalA_t_var2).max().item() for j in range(N2)]
        h_all_alphas = [np.quantile(deviations, 1 - alpha) for alpha in self.alpha_all]
        
        return h_all_alphas

    def __call__(self, score: torch.Tensor, heatmap: torch.Tensor, data: dict) -> bool:
        t = data["frame_index"].item()
        score = score.item() if isinstance(score, torch.Tensor) else score

        thresholds_t = [threshold[t].item() for threshold in self.thresholds]

        return [score > thr for thr in thresholds_t], thresholds_t
