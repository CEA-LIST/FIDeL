import torch
import numpy as np
import random


class ConformalPredictionSpaceAll:
    def __init__(self, cfg, scores: torch.Tensor, heatmaps):
        """
        Implémente la méthode de Conformal Prediction spatiale avec choix de scalA(t)
        !!! /!\ Ne marche que si encodeur de type DinoV2 qui output des features par patch !!!

        Args:
            cfg: config contenant N (utilisé si scalA(t) est locale)
            scores: [nb_episodes, T]
            heatmaps: [nb_episodes, T, nb_patch]
        """

        self.cfg = cfg
        self.alpha_all = cfg.threshold.alpha_all
        self.alpha_default = cfg.threshold.alpha_default
        self.split_rate = cfg.threshold.split_rate
        self.patch_percentage_all = cfg.threshold.percentage_patch_detection_all
        self.patch_percentage_default = cfg.threshold.percentage_patch_detection_default
        self.scal_variant = cfg.threshold.scal_variant
        self.pool_over_time = cfg.threshold.pool_over_time
        self.seed = cfg.threshold.seed
        self.eps = cfg.threshold.eps

        self.nb_episodes, self.T, self.nb_patch = heatmaps.shape

        # Étape 1: Split DcalA / DcalB
        self.split_datasets(heatmaps)

        # Étape 2: Moyenne μp sur les patches
        self.mu_p = self.DcalA.mean(dim=(0, 1))

        self.device = heatmaps.device
        self.mu_p = self.mu_p.to(self.device)

        # Étape 3: scalA(t)
        self.scalA_p_var1 = self.compute_scalA_std()
        self.scalA_p_var2 = self.compute_scalA_mad()

        # Étape 4: calcul du h et du seuil final ηₜ
        self.h_all_alphas_scalvar1 = self.compute_quantile_h_scalvar1()
        self.h_all_alphas_scalvar2 = self.compute_quantile_h_scalvar2()

        self.thresholds = []
        self.param_list = []

        # Variante 1 : patch_percentage_all, alpha = alpha_default
        for patch_percentage in self.patch_percentage_all:
            for scal_variant, scalA_p in [
                ("variant1", self.scalA_p_var1),
                # ("variant2", self.scalA_p_var2),
            ]:
                h = self.get_h(self.alpha_default, scal_variant)
                threshold = self.mu_p + h * scalA_p
                self.thresholds.append(threshold)
                self.param_list.append(
                    {
                        "alpha": self.alpha_default,
                        "split_rate": self.split_rate,
                        "scal_variant": scal_variant,
                        "patch_percentage": patch_percentage,
                    }
                )

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
        scalA(p) = 1 / nb_patch (constante pour tous les p)
        """
        return torch.ones(self.nb_patch, device=self.device) / self.nb_patch

    def compute_scalA_variant2(self):
        """
        scalA(p) = max_k |DMₖp - μp| sur DcalA
        """
        abs_dev = torch.abs(self.DcalA - self.mu_p)  # [N1, T, nb_patch]
        abs_dev = abs_dev.max(dim=0).values  # [T, nb_patch]
        return abs_dev.max(dim=0).values.to(self.device)  # [nb_patch]

    def compute_scalA_std(self) -> torch.Tensor:
        return self.DcalA.std(dim=0, unbiased=False)  # [T, nb_patch]

    def compute_scalA_mad(self) -> torch.Tensor:
        # 1.4826 * MAD (robuste)
        abs_dev = (self.DcalA - self.mu_p).abs()  # [N1, T, nb_patch]
        med = abs_dev.median(dim=0).values  # [T, nb_patch]
        return 1.4826 * med  # [T, nb_patch]

    def _pool_over_time(self, x: torch.Tensor) -> float:
        # x: [T] déviations normalisées positives pour un épisode
        if self.pool_over_time == "max":
            return float(x.max().item())
        elif self.pool_over_time == "q95":
            q = torch.quantile(x, 0.95)
            return float(q.item())
        else:  # q90
            q = torch.quantile(x, 0.90)
            return float(q.item())

    def compute_quantile_h_scalvar1(self):
        """Calcule h comme le quantile (1-alpha) des agrégations temporelles des
        déviations normalisées positives sur DcalB (upper-tail CP)."""
        N2 = self.DcalB.shape[0]
        deviations = []
        for j in range(N2):
            # déviation normalisée (one-sided, on ne garde que la partie positive)
            dev_j_t = (self.DcalB[j] - self.mu_p) / self.scalA_p_var1  # [T]
            dev_j_t = torch.clamp(dev_j_t, min=0.0)  # positive part only
            Dj = self._pool_over_time(dev_j_t)  # scalaire par épisode
            deviations.append(Dj)
        if len(deviations) == 0:
            # pas de DcalB => fallback très conservateur
            return 10.0

        h_all = [
            float(np.quantile(np.asarray(deviations, dtype=float), 1 - alpha))
            for alpha in self.alpha_all
        ]

        return [max(h, 0.0) for h in h_all]

    def compute_quantile_h_scalvar2(self):
        """Calcule h comme le quantile (1-alpha) des agrégations temporelles des
        déviations normalisées positives sur DcalB (upper-tail CP)."""
        N2 = self.DcalB.shape[0]
        deviations = []
        for j in range(N2):
            # déviation normalisée (one-sided, on ne garde que la partie positive)
            dev_j_t = (self.DcalB[j] - self.mu_p) / self.scalA_p_var2  # [T]
            dev_j_t = torch.clamp(dev_j_t, min=0.0)  # positive part only
            Dj = self._pool_over_time(dev_j_t)  # scalaire par épisode
            deviations.append(Dj)
        if len(deviations) == 0:
            # pas de DcalB => fallback très conservateur
            return 10.0

        h_all = [
            float(np.quantile(np.asarray(deviations, dtype=float), 1 - alpha))
            for alpha in self.alpha_all
        ]

        return [max(h, 0.0) for h in h_all]

    def get_h(self, alpha, scal_variant):
        if scal_variant == "variant1":
            h_list = self.h_all_alphas_scalvar1
        elif scal_variant == "variant2":
            h_list = self.h_all_alphas_scalvar2
        else:
            raise ValueError(f"Unknown scal_variant: {scal_variant}")

        # On cherche la valeur de h correspondant à alpha
        try:
            idx = self.alpha_all.index(alpha)
        except ValueError:
            # alpha = alpha_default (pas dans alpha_all)
            return np.quantile(
                [
                    (
                        self.mu_p
                        - (self.DcalB[j])
                        / (getattr(self, f"scalA_p_{scal_variant}") + 1e-6)
                    )
                    .max()
                    .item()
                    for j in range(self.DcalB.shape[0])
                ],
                1 - alpha,
            )
        return h_list[idx]

    def predict_frame(self, heatmap_t: float, t: int) -> bool:
        # Comme tu l'as confirmé: plus c'est grand => plus c'est anormal
        return float(heatmap_t) > self.threshold_at(t)

    def has_at_least_X_percent_true(self, tensor):
        total = tensor.numel()  # nombre total d'éléments
        n_true = tensor.sum().item()  # nombre de True (puisqu'ils valent 1)
        return n_true / total >= self.patch_percentage

    def __call__(
        self, score: torch.Tensor, heatmap: torch.Tensor, data: dict = None
    ) -> bool:
        anomaly_flags = []

        t = (
            int(data["frame_index"])
            if not hasattr(data["frame_index"], "item")
            else int(data["frame_index"].item())
        )
        thresholds_t = []
        for threshold, params in zip(self.thresholds, self.param_list):
            patch_percentage = params["patch_percentage"]
            anomaly_flag = heatmap.squeeze() >= threshold[t, :]  # [nb_patch]
            thresholds_t.append(threshold[t])
            total = anomaly_flag.numel()
            n_true = anomaly_flag.sum().item()
            is_anomalous = (n_true / total) >= patch_percentage
            anomaly_flags.append(is_anomalous)

        return anomaly_flags, self.thresholds  # liste de booléens
