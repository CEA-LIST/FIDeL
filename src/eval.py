from pathlib import Path
import hydra
import torch
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("lerobot").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
from torch.utils.data import DataLoader
from einops import rearrange
import os
from tqdm import tqdm
from omegaconf import OmegaConf

# from logger import Logger

from lerobot.datasets.factory import make_dataset

from dataset_eval import Safe_imitation
from failure_detection_and_recovery import FailureDetector
from memory_usage import get_cpu_memory_usage, get_gpu_memory_usage


class Eval_With_Lerobot:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.ad = cfg.anomaly_detection.name
        self.process = "eval"

        # load evaluation dataset
        print("\r\n *********** Loading evaluation dataset ***********")
        print("dataset path :", self.cfg.eval_dataset_repo_id)
        lerobot_dataset = make_dataset(self.cfg, self.cfg.eval_dataset_repo_id)
        self.total_nb_frame = lerobot_dataset.num_frames
        self.nb_episode = lerobot_dataset.num_episodes
        self.data = lerobot_dataset.features
        self.dataloader = DataLoader(lerobot_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # récupérer le nombre d'épisodes / nb de frames par épisode / identité des capteurs

        # Initiate safe imitation
        self.safe_imitation = Safe_imitation(cfg, eval_real_robot=False)
        # Initiate threshold
        self.failure_detector = FailureDetector(cfg)

        self.action_history = []
        self.image_history = []

    def eval_ad(self):
        idx = 0
        score_list = []
        anomaly_list = []
        failure_list = []
        threshold_list = []
        print("\r\n************* Starting Evaluation **************\r\n")

        self.safe_imitation.setup(self.dataloader)  # Loading Trained AD Model
        (
            expert_score,
            expert_heatmap,
            global_mean_expert_data,
            global_std_expert_data,
        ) = self.safe_imitation.get_expert_score()  # computation over the expert data to normalize score
        self.failure_detector.threshold_calibration(
            expert_heatmap,
            expert_score,
            global_mean_expert_data,
            global_std_expert_data,
        )
        self.param_list = self.failure_detector.Anomaly_jury.param_list

        pbar = tqdm(
            total=self.total_nb_frame,
            desc="score computation",
            unit="frame",
            leave=True,
        )
        score = 0

        for data in self.dataloader:
            if not idx % 1:  # to chose the frequency of score computation
                # data: dict_keys(['observation.images.logitech', 'action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'])

                # compute the anomaly score
                score, heatmap = self.safe_imitation.eval_ad(data)

                score_list.append(score)

                anomaly_value, vlm_prediction, threshold_value = self.failure_detector.anomaly_threshold(score, heatmap, data)
                #print("anomaly:", anomaly_value)
                #print("vlm response:", vlm_prediction)
                anomaly_list.append(anomaly_value)
                failure_list.append(vlm_prediction)
                threshold_list.append(threshold_value)

            cpu_usage = get_cpu_memory_usage()
            gpu_usage = get_gpu_memory_usage()
            pbar.set_description(
                f"{((pbar.n + 1) / self.total_nb_frame) * 100:5.1f}% | CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}% | score: {score:.4f}"
            )
            pbar.update(1)
            idx += 1

        # dowload score_list
        score_list = torch.tensor(score_list)

        if isinstance(anomaly_list[0], torch.Tensor):
            anomaly_list = torch.stack(anomaly_list)
        else:
            anomaly_list = torch.tensor(anomaly_list)

        if isinstance(threshold_list[0], torch.Tensor):
            threshold_list = torch.stack(threshold_list)
        else:
            threshold_list = torch.tensor(threshold_list)

        if isinstance(failure_list[0], torch.Tensor):
            failure_list = torch.stack(failure_list)
        else:
            failure_list = torch.tensor(failure_list)

        if self.ad == "Representation":
            file_path = (
                self.cfg.result_dir
                + "/score"
                + f"/score_{self.cfg.encoder}_{self.cfg.eval_dataset_id}_{self.cfg.anomaly_detection.name}_{self.cfg.anomaly_detection.distance_type}_{self.cfg.threshold.name}.pt"
            )
        else:
            file_path = (
                self.cfg.result_dir
                + "/score"
                + f"/score_{self.cfg.encoder}_{self.cfg.eval_dataset_id}_{self.cfg.anomaly_detection.name}_{self.cfg.threshold.name}.pt"
            )

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(
            {
                "eval_score": score_list,
                "anomaly_values": anomaly_list,
                "safety_values": failure_list,
                "param_list": self.param_list,
                "threshold_values": threshold_list,
            },
            file_path,
        )

        print(f"scores have been successfully stored in : {file_path}\r\n")
        print("*********** Evaluation over ***********")


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    from eval import Eval_With_Lerobot as E

    workspace = E(cfg)

    workspace.eval_ad()


if __name__ == "__main__":
    main()
