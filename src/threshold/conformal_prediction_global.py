import torch
import numpy as np

class ConformalPredictionGlobal:
    def __init__(self, cfg, scores: torch.Tensor):
        """
        Implements a Global Conformal Prediction method (time-invariant).
        Instead of seeking joint coverage over the entire trajectory (which explodes with .max()),
        this method guarantees marginal coverage: (1 - alpha)% of normal frames will be below the threshold.
        
        Args:
            cfg: config
            scores: [nb_episodes, T] tensor of expert scores
        """
        self.cfg = cfg
        self.alpha_all = cfg.threshold.alpha_all
        self.split_rate = cfg.threshold.split_rate
        
        # Flatten all expert scores, because the Representation score is time-invariant
        all_expert_scores = scores.flatten().cpu().numpy()
        
        self.thresholds = []
        self.param_list = []
        
        # For each confidence level alpha, the threshold is simply the (1 - alpha) quantile
        for alpha in self.alpha_all:
            threshold = np.quantile(all_expert_scores, 1 - alpha)
            self.thresholds.append(threshold)
            self.param_list.append({
                "alpha": alpha,
                "split_rate": self.split_rate,
                "type": "global_marginal"
            })
            
    def __call__(self, score: torch.Tensor, heatmap: torch.Tensor, data: dict) -> bool:
        score = score.item() if isinstance(score, torch.Tensor) else score
        # Compare the score to the global threshold (independent of t)
        return [score > thr for thr in self.thresholds], self.thresholds
