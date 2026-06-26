import os
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
from torchvision import models, transforms
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
)

from dataset.data_processing import CustomDataset
from dataset.encoder_lerobot import (
    DinoV2_extract,
    Resnet18_extract,
    SigLIP_extract,
)
from lerobot.datasets.factory import make_dataset
from anomaly_detection.Representation import representation
from memory_usage import get_cpu_memory_usage, get_gpu_memory_usage


class Train_with_Lerobot:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"Workspace: {self.work_dir}")
        self.cfg = cfg
        # utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.ad = cfg.anomaly_detection.name
        self.network = None
        self.feature_size = None
        self.action_size = None
        self.criterion = None
        self.sample_length = cfg.sample_length
        self.patch_size = 0
        self.nb_of_frame_per_episode = 0
        self.nb_of_frame_max = None
        self.nb_of_episode = 0
        self.pixel_values_eval = None
        self.is_trained = False
        self.process = "train"

        # Loading models once and for all
        print("\n*********** Loading encoder model ***********")

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

            # bnb_config = BitsAndBytesConfig(load_in_4bit=True)
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
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        print("\n")

    def process_data_lerobot(self):
        features_dir = Path(self.cfg.encoder_features_dir)
        folder_name = f"features_{self.cfg.encoder}_{self.cfg.train_dataset_id}"
        features_folder = features_dir / folder_name

        # Checking if the features have already been extracted
        if features_folder.exists() and features_folder.is_dir():
            files = os.listdir(features_folder)  # List of files in the directory
            if len(files) > 0:
                print("Features have already been extracted.")
                # Retrieve these features and return the dataloader

                data_dict = torch.load(
                    features_folder
                    / files[
                        0
                    ],  # Take the first file, normally there should be just one
                    weights_only=False,
                )  # TODO: Check the size

                self.feature_size = data_dict["features"][0].shape[-1]

                if self.cfg.encoder == "dinoV2":
                    self.patch_size = data_dict["features"].shape[-2]
                if self.cfg.encoder in ["resnet18", "SigLIP"]:
                    self.patch_size = 1

                self.action_size = data_dict["action"].shape[-1]
                self.nb_of_frame_per_episode = int(data_dict["frame_index"][-1] + 1)
                self.nb_of_frame_max = int(max(data_dict["frame_index"]) + 1)
                self.nb_of_episode = int(data_dict["episode_index"][-1] + 1)

                return data_dict

        print("\n*********** Loading Demonstration Data ************")
        lerobot_dataset = make_dataset(self.cfg, self.cfg.train_dataset_repo_id)
        # data: dict_keys(['observation.images.logitech', 'action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'])

        print("\n")
        print(
            f"********** Proceeding to features Extraction with {self.cfg.encoder} ... **********"
        )
        print("\n")

        if self.cfg.encoder == "dinoV2":
            data_dict, self.pixel_values_eval = DinoV2_extract(
                lerobot_dataset,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_dict["features"].shape[-1]
            self.action_size = data_dict["action"].shape[-1]
            self.nb_of_frame_per_episode = int(data_dict["frame_index"][-1] + 1)
            self.nb_of_frame_max = int(max(data_dict["frame_index"]) + 1)
            self.nb_of_episode = int(data_dict["episode_index"][-1] + 1)
            self.patch_size = data_dict["features"].shape[-2]

        elif self.cfg.encoder == "SigLIP":
            data_dict, self.pixel_values_eval = SigLIP_extract(
                lerobot_dataset,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_dict["features"].shape[-1]
            self.action_size = data_dict["action"].shape[-1]
            self.nb_of_frame_per_episode = int(data_dict["frame_index"][-1] + 1)
            self.nb_of_frame_max = int(max(data_dict["frame_index"]) + 1)
            self.nb_of_episode = int(data_dict["episode_index"][-1] + 1)
            self.patch_size = 1

        elif self.cfg.encoder == "resnet18":
            data_dict, self.pixel_values_eval = Resnet18_extract(
                lerobot_dataset,
                self.device,
                self.process,
                self.cfg,
                self.encoder_model,
                self.encoder_processor,
            )
            self.feature_size = data_dict["features"].shape[-1]
            self.action_size = data_dict["action"].shape[-1]
            self.nb_of_frame_per_episode = int(data_dict["frame_index"][-1] + 1)
            self.nb_of_frame_max = int(max(data_dict["frame_index"]) + 1)
            self.nb_of_episode = int(data_dict["episode_index"][-1] + 1)
            self.patch_size = 1

        else:
            raise NotImplementedError

        print("\nFeatures extraction complete")
        torch.cuda.empty_cache()

        return data_dict

    def train_ad(self):
        weights_dir = (
            Path(self.cfg.weights_dir)
            / f"{self.ad}_{self.cfg.encoder}_{self.cfg.train_dataset_id}"
        )

        # Checking if the model has already been trained
        if weights_dir.exists() and weights_dir.is_dir():
            files = os.listdir(weights_dir)  # List of files in the directory
            if len(files) > 0:
                self.is_trained = True
                print("\nTraining has already been done.")
                return 0  # Exit function since training is already complete

        data_dict = self.process_data_lerobot()
        train_dict = data_dict

        dataset_train = CustomDataset(train_dict, self.cfg.sample_length)
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.cfg.anomaly_detection.batch_size,
            shuffle=False,
        )

        ###################### Representation - Aggregating expert data ######################

        if self.ad == "Representation":
            self.network, self.criterion, optimizer = representation(
                nb_of_episodes=self.nb_of_episode,
                frame_per_episode=self.nb_of_frame_max,
                patch_dim=self.patch_size,
                features_dim=self.feature_size,
                cfg=self.cfg,
                training=True,
            )

            self.network = self.network.to(self.device)
            self.network.train()

            print(
                "\n************** Aggregating demonstration data into memory ... **************\n"
            )

            pbar = tqdm(
                total=self.nb_of_episode,
                desc="Mean computation",
                unit="episode",
                leave=True,
            )

            # We first compute the mean over episodes of normal features, then we compute the std,
            # in order to avoid storing all features of all episodes in memory at the same time (which could be too large).

            # ====== Mean computation ======
            for data in dataloader_train:
                _ = self.network(data)

                cpu_usage = get_cpu_memory_usage()
                gpu_usage = get_gpu_memory_usage()
                pbar.set_description(
                    f"{((pbar.n + 1) / self.nb_of_episode) * 100:5.1f}% | CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}%"
                )
                pbar.update(1)

            pbar.close()

            pbar = tqdm(
                total=self.nb_of_episode,
                desc="Std computation",
                unit="episode",
                leave=True,
            )

            self.network.memory.compute_mean()
            self.network.memory.counter.zero_()
            self.network.memory.flag_avg_has_been_computed = True

            # ====== Std computation ======
            for data in dataloader_train:
                _ = self.network(data)

                cpu_usage = get_cpu_memory_usage()
                gpu_usage = get_gpu_memory_usage()
                pbar.set_description(
                    f"{((pbar.n + 1) / self.nb_of_episode) * 100:5.1f}% | CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}%"
                )
                pbar.update(1)

            pbar.close()

            self.network.memory.compute_std()

            print("\nStoring complete.\n")

            weights_dir.mkdir(parents=True, exist_ok=True)

            # Using Pathlib to build safe paths
            final_memory_path = weights_dir / f"network_memory_{self.ad}_final.pth"

            mean_memory, std_memory = self.network.memory.get_mean_and_std()
            print("Memory stats shapes:", mean_memory.shape, std_memory.shape)

            # Save the mean and std over episodes of normal features
            torch.save(
                {
                    "mean_memory": mean_memory,
                    "std_memory": std_memory,
                },
                final_memory_path,
            )

            print(f"Memory stats have been saved in: {final_memory_path}")

        self.is_trained = True
        torch.cuda.empty_cache()


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3.2")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))

    # Keeping imports local as defined in original script
    from store_memory_data import Train_with_Lerobot as T

    workspace = T(cfg)
    workspace.train_ad()

    torch.cuda.empty_cache()  # Freeing GPU memory


if __name__ == "__main__":
    main()
