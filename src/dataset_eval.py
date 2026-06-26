

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision import models, transforms
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
)

from dataset.encoder_lerobot import (
    DinoV2_extract,
    SigLIP_extract,
    Resnet18_extract,
)
from anomaly_detection.Representation import representation
from anomaly_detection.score_computation import compute_score
from dataset.data_processing import CustomDataset
from memory_usage import get_cpu_memory_usage, get_gpu_memory_usage


class Safe_imitation:
    def __init__(self, cfg, eval_real_robot):
        self.work_dir = Path.cwd()
        print(f"Workspace: {self.work_dir}")
        self.cfg = cfg
        self.ad = self.cfg.anomaly_detection.name

        if eval_real_robot:
            self.cfg = cfg.safe_imitation
            self.robot_cfg = cfg
            self.ad = self.cfg.ad_type

        # utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"Anomaly Detection type: {self.ad}")
        print(f"Encoder type: {self.cfg.encoder}")

        self.network = None
        self.feature_size = None
        self.patch_size = None
        self.sample_length = self.cfg.sample_length
        self.pixel_values_eval = None
        self.process = "eval"
        self.mean_expert_score, self.std_expert_score = None, None
        self.first_loop_flag = True
        self.last_action = None
        self.last_img_feature = None
        self.last_episode_idx = None
        self.index_heatmap = 0
        self.inference_time = []
        self.min_len = None

        if self.cfg.encoder == "dinoV2":
            self.encoder_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", use_fast=False
            )
            self.encoder_model = AutoModel.from_pretrained("facebook/dinov2-base").to(
                self.device
            )

        if self.cfg.encoder == "SigLIP":
            self.encoder_processor = AutoProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            self.encoder_model = AutoModel.from_pretrained(
                "google/siglip-base-patch16-224",
                # quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",
            ).to(self.device)

        if self.cfg.encoder == "resnet18":
            # Using weights="DEFAULT" to avoid PyTorch deprecation warnings on pretrained=True
            resnet18 = models.resnet18(weights="DEFAULT")
            resnet18.fc = torch.nn.Identity()
            self.encoder_model = resnet18.to(self.device)
            self.encoder_processor = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def process_data_inference(self, data):
        # TODO: check if the inference data type matches the training data type --> number of cameras, name of the camera, shape of the image
        # ....
        # Extract data
        # TODO: uncomment the next line when the previous TODO is done
        # observations = [[obs[f'observation.images.{camera_name}'] for camera_name in list(self.cfg.cameras.keys())] for obs in observation_hist]

        observations = data[f"observation.images.{self.cfg.camera}"]
        action = data["action"]
        self.action_size = data["action"].shape[-1]

        # Convert raw data into features with the selected encoder
        if self.cfg.encoder == "dinoV2":
            data_features, self.pixel_values_eval = DinoV2_extract(
                observations,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_features.shape[-1]
            self.patch_size = data_features.shape[-2]

        elif self.cfg.encoder == "SigLIP":
            data_features, self.pixel_values_eval = SigLIP_extract(
                observations,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_features[0].shape[-1]
            self.patch_size = 1

        elif self.cfg.encoder == "resnet18":
            data_features, self.pixel_values_eval = Resnet18_extract(
                observations,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_features[0].shape[-1]
            self.patch_size = 1

        return data_features, observations, action

    def truncate_lists_to_tensor(self, list_of_lists):
        min_len = min(len(lst) for lst in list_of_lists)
        self.min_len = min_len

        truncated = []
        for lst in list_of_lists:
            # Convert elements to tensors if not already
            tensors = [
                item.clone().detach()
                if isinstance(item, torch.Tensor)
                else torch.tensor(item)
                for item in lst[:min_len]
            ]
            truncated.append(torch.stack(tensors))

        # Stack directly outputs a tensor, avoiding the "torch.tensor(tensor)" warning
        truncated = torch.stack(truncated).squeeze()
        return truncated

    def get_expert_score(self):
        """Goal: retrieve the mean and std of the anomaly score over the training data to extract the offset."""

        print("\r\nComputing mean and std over expert score...")

        # Check if the mean and std have already been computed for this cfg
        score_dir = Path(self.cfg.weights_dir) / f"{self.ad}_{self.cfg.encoder}_{self.cfg.train_dataset_id}"

        if self.ad == "Representation":
            expert_score_file_path = (
                score_dir
                / f"expert_score_{self.cfg.encoder}_{self.cfg.train_dataset_id}_{self.cfg.anomaly_detection.name}_{self.cfg.anomaly_detection.distance_type}.pt"
            )
        else:
            expert_score_file_path = (
                score_dir
                / f"expert_score_{self.cfg.encoder}_{self.cfg.train_dataset_id}_{self.cfg.anomaly_detection.name}.pt"
            )

        if expert_score_file_path.exists():
            loaded_data = torch.load(expert_score_file_path, weights_only=False)
            expert_heatmap = loaded_data["expert_heatmaps"]
            expert_score = loaded_data["expert_scores"]
            mean_expert_score = loaded_data["mean"]
            std_expert_score = loaded_data["std"]
            self.min_len = loaded_data["min_len"]

        else:
            # Check if expert data features have already been extracted
            features_dir = Path(self.cfg.encoder_features_dir)
            folder_name = f"features_{self.cfg.encoder}_{self.cfg.train_dataset_id}"
            folder_path = features_dir / folder_name

            if folder_path.exists() and folder_path.is_dir():
                files = os.listdir(folder_path)  # List of files in the folder
                if len(files) > 0:
                    print("\r\nExpert features have been extracted.\r\n")
                else:
                    print(f"The expert features are not at: {folder_path}")
                    raise FileNotFoundError

            else:
                print(f"The expert features are not at: {folder_path}")
                raise FileNotFoundError

            # Compute the score over all features
            expert_score = []
            expert_heatmap = []
            episode_expert_score = []
            episode_expert_heatmap = []

            network = self.network.to(self.device)
            network.eval()

            data = torch.load(
                folder_path / "features.pt",
                weights_only=False,
            )
            episode_nb = int(min(data["episode_index"]))  # Init of episode idx
            dataset_train = CustomDataset(data, self.cfg.sample_length)

            total_frames = len(dataset_train)
            pbar = tqdm(
                total=total_frames,
                desc="Mean computation",
                unit="frame",
                leave=True,
            )

            for batch_idx, frame_data in enumerate(dataset_train):
                # Append action data by concatenation to the feature data if required
                if self.cfg.use_action:
                    frame_data["features"] = torch.cat(
                        [frame_data["features"], frame_data["action"]], dim=0
                    )

                output = self.network(frame_data)
                score = compute_score(
                    self,
                    output,
                    frame_data,
                )

                new_episode_nb = frame_data["episode_index"].item()
                if episode_nb != new_episode_nb:  # One sub-list per episode
                    expert_score.append(episode_expert_score)
                    if self.cfg.encoder == "dinoV2":
                        expert_heatmap.append(episode_expert_heatmap)
                    episode_expert_score = []
                    episode_expert_heatmap = []
                    episode_nb = new_episode_nb

                if self.ad != "Representation":
                    episode_expert_score.append(score)
                else:
                    episode_expert_score.append(score[0])
                    episode_expert_heatmap.append(score[1])

                cpu_usage = get_cpu_memory_usage()
                gpu_usage = get_gpu_memory_usage()
                pbar.set_description(
                    f"{((pbar.n + 1) / total_frames) * 100:5.1f}% | CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}%"
                )
                pbar.update(1)

            # Store the last episode
            expert_score.append(episode_expert_score)
            expert_heatmap.append(episode_expert_heatmap)

            expert_score = self.truncate_lists_to_tensor(expert_score)
            if self.cfg.encoder == "dinoV2":
                expert_heatmap = self.truncate_lists_to_tensor(expert_heatmap)
            else:
                expert_heatmap = None

            mean_expert_score = expert_score.mean()
            std_expert_score = expert_score.std()

            # Save in a torch file
            score_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "expert_scores": expert_score,
                    "expert_heatmaps": expert_heatmap,
                    "mean": mean_expert_score,
                    "std": std_expert_score,
                    "min_len": self.min_len,
                },
                expert_score_file_path,
            )

        print(
            f"Computing score complete, mean_expert_score = {mean_expert_score}, std_expert_score = {std_expert_score} \r\n"
        )

        self.mean_expert_score = mean_expert_score
        self.std_expert_score = std_expert_score
        torch.cuda.empty_cache()
        return expert_score, expert_heatmap, mean_expert_score, std_expert_score

    def display_img(
        self,
        images,
        score_ad,
        features_scores,
    ):
        """Display image without gradcam for Representation"""
        img_score_folder_path = (
            Path(self.cfg.result_dir)
            / "img_with_score"
            / f"{self.cfg.eval_dataset_id}_{self.ad}_{self.cfg.anomaly_detection.distance_type}"
        )
        img_score_folder_path.mkdir(parents=True, exist_ok=True)

        output_path = img_score_folder_path / f"image_{self.index_heatmap}.png"
        self.index_heatmap += 1  # Increment to save subsequent images

        img = images[-1]

        # Rearrange the image shape
        img = rearrange(img.squeeze(), "c h w -> h w c").cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Reshape features_scores to (16,16) and resize to the image size
        heatmap = features_scores.view(16, 16).detach().cpu().numpy()
        heatmap = cv2.resize(
            heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
        )

        # Normalize the heatmap
        min_val = 3
        max_val = 60
        heatmap = (heatmap - min_val) / (max_val - min_val)
        heatmap = np.clip(heatmap * 255, 0, 255)

        # Apply a colormap to visualize the scores
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # Merge the image with the heatmap
        alpha = 0.5
        superimposed_img = cv2.addWeighted(img, 1, heatmap_colored, alpha, 0)

        # Add the anomaly score in red in the top-left corner
        text = f"Score: {(score_ad / 256):.2f}"
        cv2.putText(
            superimposed_img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # Save the image with the superimposed heatmap
        cv2.imwrite(str(output_path), superimposed_img)

    def setup(self, dataloader):
        weights_dir = (
            Path(self.cfg.weights_dir)
            / f"{self.ad}_{self.cfg.encoder}_{self.cfg.train_dataset_id}"
        )

        # Checking if the expert data has already been extracted and stored in the right folder
        if weights_dir.exists() and weights_dir.is_dir():
            files = os.listdir(weights_dir)  # List of files in the folder
            if len(files) > 0:
                self.is_trained = True
                print(
                    f"Demonstration data has already been extracted and stored in {weights_dir}"
                )
        else:
            print(f"The demonstration data is not at: {weights_dir}")
            # TODO: maybe ask the user if he wants to launch training, with which AD type?
            # Or proceed to inference without Safe
            raise FileNotFoundError  # Training has not been done yet

        if self.ad == "Representation":
            final_memory_path = weights_dir / f"network_memory_{self.ad}_final.pth"
            memory_dict = torch.load(final_memory_path, weights_only=False)

            self.network, self.criterion, _ = representation(
                nb_of_episodes=0,
                frame_per_episode=0,
                patch_dim=0,
                features_dim=0,
                cfg=self.cfg,
                training=False,
            )

            self.network.memory.memory_avg = memory_dict["mean_memory"]
            self.network.memory.memory_std = memory_dict["std_memory"]
            self.network.memory.cfg.anomaly_detection.distance_type = (
                self.cfg.anomaly_detection.distance_type
            )

        else:
            raise NotImplementedError

    def eval_ad(self, data):
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            start_time = time.time()
    
            heatmap = None
            features, images, action = self.process_data_inference(data)
    
            ############################## Evaluation ##############################
    
            self.network.to(self.device)
            self.network.eval()
            if hasattr(self, 'encoder_model') and self.encoder_model is not None:
                self.encoder_model.eval()
            features = features.squeeze()
            action = action.squeeze().to(self.device)
    
            if self.cfg.sample_length > 1:
                if self.last_episode_idx != data["episode_index"]:
                    input = torch.cat([features, features])
                    if self.cfg.use_action:
                        input = torch.cat([input, action, action])
    
                else:
                    input = torch.cat([features, self.last_img_feature])
                    # Append action data by concatenation to the feature data
                    if self.cfg.use_action:
                        input = torch.cat([input, action, self.last_action])
    
            else:
                input = features
    
            input = input.to(self.device)
            output = self.network(input.float())
    
            if self.ad == "Representation":
                heatmap = output[1]
                output = output[0]

            inference_time_t = time.time() - start_time
            self.inference_time.append(inference_time_t)
    
            score_ad = compute_score(self, output, input)
            if self.std_expert_score == 0:
                self.std_expert_score = 1
    
            # Saving history
            self.last_img_feature = features
            self.last_action = action
            self.last_episode_idx = data["episode_index"]
    
            # Save heatmap
            if self.cfg.display_img_with_score:
                self.display_img(
                    images,
                    score_ad,
                    heatmap,
                )
    
            return score_ad, heatmap
