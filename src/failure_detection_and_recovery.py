import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from einops import rearrange
from torchvision.transforms.functional import to_pil_image

from VLM_qwen import VLM_SemanticFilter
# from VLM_gpt import VLM_SemanticFilter

from threshold.simple_gaussian import Simple_gaussian
from threshold.multi_gaussian_time import Multi_gaussian_time
from threshold.conformal_prediction_time import ConformalPredictionTime
from threshold.conformal_prediction_time_2 import ConformalPredictionTime2
from threshold.conformal_prediction_time_all import ConformalPredictionTimeAll
from threshold.conformal_prediction_global import ConformalPredictionGlobal
from threshold.conformal_prediction_space import ConformalPredictionSpace
from threshold.conformal_prediction_space_all import ConformalPredictionSpaceAll
from threshold.conformal_prediction_space_all_2 import ConformalPredictionSpaceAll2
from threshold.conformal_prediction_time_and_space import (
    ConformalPredictionTimeAndSpace,
)

from lerobot.datasets.factory import make_dataset


def superpose_heatmap(image, matrix):
    """
    Displays the image with a superimposed heatmap for representation.
    """
    # Rearrange the image from (C, H, W) to (H, W, C)
    img = rearrange(image.squeeze(), "c h w -> h w c").cpu().numpy()
    img = (img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Reshape matrix to (16, 16) and resize to the image size
    heatmap = matrix.view(16, 16).cpu().detach().numpy()
    heatmap = cv2.resize(
        heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
    )

    # Normalize the heatmap
    min_val = 3
    max_val = 60
    heatmap = (heatmap - min_val) / (max_val - min_val)

    threshold = 0.65
    heatmap = np.where(heatmap < threshold, 0, heatmap)
    heatmap = np.clip(heatmap * 255, 0, 255)

    # Apply a colormap to visualize the scores
    heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # Merge the image with the heatmap
    alpha = 0.5
    mask = heatmap > 0
    heatmap_colored[~mask] = 0
    superimposed_img = cv2.addWeighted(img, 1, heatmap_colored, alpha, 0)

    return superimposed_img


def display_images(expert, anomaly, heatmap=None):
    """
    Displays the expert image, anomaly image, and optionally the heatmap side-by-side.
    """
    plt.close("all")  # Close the previous figure
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(expert)
    plt.title("Expert")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(anomaly)
    plt.title("Anomaly")
    plt.axis("off")

    if heatmap is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Heatmap")
        plt.axis("off")

    plt.show(block=False)
    plt.pause(0.001)  # Allow time for the figure to render


class FailureDetector:
    """Threshold calibration + semantic failure detection"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.Anomaly_jury = None
        self.calibration_done = False
        self.raise_anomaly = False
        self.raise_failure = False
        self.last_vlm_prompt = ""
        self.episode = 0
        self.frame_index = 0

        print(f"Threshold type: {cfg.threshold.name}")
        if self.cfg.use_VLM:
            self.semantic_filter = VLM_SemanticFilter()
        else:
            self.semantic_filter = None

        # Import expert images for VLM
        self.lerobot_dataset = make_dataset(self.cfg, self.cfg.train_dataset_repo_id)

    def threshold_calibration(
        self, heatmaps, scores, global_mean_expert_data, global_std_expert_data
    ):
        """
        Args:
            scores: All scores for all frames of demo data, shape (nb_episodes, nb_frame_per_episode)
            heatmaps: All heatmaps for all frames of demo data, shape (nb_episodes, nb_frame_per_episode, nb_frames)
            global_mean_expert_data (float): Mean score for all demonstrations
            global_std_expert_data (float): Std score for all demonstrations
        """

        threshold_mapping = {
            "simple_gaussian": Simple_gaussian,
            "multi_gaussian_time": Multi_gaussian_time,
            "conformal_prediction_time": ConformalPredictionTime,
            "conformal_prediction_space": ConformalPredictionSpace,
            "conformal_prediction_space_all": ConformalPredictionSpaceAll,
            "conformal_prediction_space_all_2": ConformalPredictionSpaceAll2,
            "conformal_prediction_time_and_space": ConformalPredictionTimeAndSpace,
            "conformal_prediction_time_all": ConformalPredictionTimeAll,
            "conformal_prediction_time_2": ConformalPredictionTime2,
            "conformal_prediction_global": ConformalPredictionGlobal,
        }

        if self.cfg.threshold.name in threshold_mapping:
            if self.cfg.threshold.name == "simple_gaussian":
                self.Anomaly_jury = threshold_mapping[self.cfg.threshold.name](
                    self.cfg, global_mean_expert_data, global_std_expert_data
                )
            elif "space" in self.cfg.threshold.name:
                self.Anomaly_jury = threshold_mapping[self.cfg.threshold.name](
                    self.cfg, scores, heatmaps
                )
            else:
                self.Anomaly_jury = threshold_mapping[self.cfg.threshold.name](
                    self.cfg, scores
                )
            self.calibration_done = True
        else:
            raise ValueError(f"Unknown threshold type: {self.cfg.threshold.name}")

    def anomaly_threshold(self, score, heatmap, data):
        """
        Args:
            score: Value outputted by the anomaly detector for the current frame
            heatmap: Anomaly score per patch for the current frame
            data (dict): Dictionary containing observation, action, index, etc.
        """
        anomaly_binary, threshold = self.Anomaly_jury(score, heatmap, data)

        if self.cfg.use_VLM:
            # Handle real-time semantic filtering
            vlm_predictions = []
            if any(anomaly_binary):
                self.raise_anomaly = True
                vlm_prediction = self.failure_detection(score, data, heatmap)
            else:
                vlm_prediction = False

            for is_anomalous in anomaly_binary:
                vlm_predictions.append(vlm_prediction if is_anomalous else False)
        else:
            vlm_predictions = [False]

        return anomaly_binary, vlm_predictions, threshold

    def failure_detection(self, score, data, raw_heatmap):
        # Safely extract tensor values
        t = (
            data["frame_index"].item()
            if isinstance(data["frame_index"], torch.Tensor)
            else int(data["frame_index"])
        )
        n = (
            data["episode_index"].item()
            if isinstance(data["episode_index"], torch.Tensor)
            else int(data["episode_index"])
        )

        # Note: 'logitech_1' might need to be dynamic (e.g., self.cfg.camera) depending on your config setup
        image_anomaly = data["observation.images.logitech_1"]

        if self.cfg.encoder == "dinoV2":
            heatmap = raw_heatmap.view(16, 16).cpu().detach().numpy()
            heatmap = cv2.resize(
                heatmap,
                (image_anomaly.shape[1], image_anomaly.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            heatmap = torch.tensor(heatmap)

        data_expert = self.lerobot_dataset[t]
        image_expert = data_expert["observation.images.logitech_1"]
        superimposed_img = superpose_heatmap(image_anomaly, raw_heatmap)

        # If the tensor is float in [0,1], clamp it before conversion
        if image_expert.dtype == torch.float:
            image_expert = image_expert.clamp(0, 1)

        # Convert to PIL images for VLM processing
        img_pil_anomaly = to_pil_image(superimposed_img.squeeze())

        # Reset the last prompt to empty if we change episodes or if t is not consecutive
        # if n != self.episode or t != self.frame_index + 1:
        self.last_vlm_prompt = ""
        self.episode = n
        self.frame_index = t

        severity, new_prompt = self.semantic_filter.analyze(
            self.cfg,
            image_expert,
            img_pil_anomaly,
            self.cfg.prompt,
            self.last_vlm_prompt,
        )

        self.last_vlm_prompt = (
            "Finally, here is the last prompt you outputted regarding the last frame, "
            "if the anomaly is similar, stay consistent with your last decision:\n"
            + new_prompt
        )
        
        return severity
