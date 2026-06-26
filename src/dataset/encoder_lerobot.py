import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm

from memory_usage import get_cpu_memory_usage, get_gpu_memory_usage


def clear_folder(folder_path):
    """Empties a folder without deleting it."""
    try:
        shutil.rmtree(folder_path)  # Deletes the folder and all its contents
        os.makedirs(folder_path)  # Recreates the folder empty
    except Exception as e:
        print(f"Error clearing folder: {e}")


#############################################################################################
#################################### Features Extraction ####################################
#############################################################################################


def save_features(data_dict, save_dir):
    """
    Saves the dictionary containing data and extracted features for each episode into a .pt file.

    Args:
        data_dict (dict): Dictionary containing features and metadata.
        save_dir (str or Path): Path to the directory where features will be saved.
    """
    save_dir = Path(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data_dict, save_dir)


def model_extract_features_train(
    cfg,
    dataset,
    processor,
    transform,
    model,
    device,
    dino_flag,
    sig_flag,
    batch_size=64,
):
    """
    Processes images in batches using DataLoader, ensuring metadata and features
    are properly extracted and aligned.

    Args:
        cfg: Configuration object.
        dataset: Dataset of dictionaries in Lerobot format.
        processor: Preprocessor (e.g., AutoImageProcessor from transformers).
        transform: Torchvision transforms (used for ResNet).
        model: Feature extraction model.
        device: Target device (GPU or CPU).
        dino_flag (bool): True if using DINOv2.
        sig_flag (bool): True if using SigLIP.
        batch_size (int): Size of the batch.

    Returns:
        tuple: (data_dict, inputs) where data_dict contains concatenated features and metadata.
    """

    # Prompt for SigLIP
    instructions = cfg.prompt
    prompt = f"What action should the robot take to {instructions}?"

    # Calculate total number of batches for the progress bar
    total_frames = dataset.num_frames
    total_batches = total_frames // batch_size + (1 if total_frames % batch_size != 0 else 0)
    pbar = tqdm(
        total=total_batches, desc="Extracting features", unit="batch", leave=True
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    actions = None
    timestamps = None
    frame_indices = None
    episode_indices = None
    indices = None
    task_indices = None
    features = None

    current_idx = 0

    for batch in dataloader:
        batch_len = batch["action"].shape[0]

        # Pre-allocate tensors on the first batch to avoid MemoryError during torch.cat
        if actions is None:
            actions = torch.empty((total_frames, *batch["action"].shape[1:]), dtype=torch.float32)
            timestamps = torch.empty((total_frames, 1), dtype=torch.float32)
            frame_indices = torch.empty((total_frames, 1), dtype=torch.float32)
            episode_indices = torch.empty((total_frames, 1), dtype=torch.float32)
            indices = torch.empty((total_frames, 1), dtype=torch.float32)
            task_indices = torch.empty((total_frames, 1), dtype=torch.float32)

        actions[current_idx:current_idx + batch_len] = batch["action"].cpu()
        timestamps[current_idx:current_idx + batch_len] = batch["timestamp"].unsqueeze(1).cpu()
        frame_indices[current_idx:current_idx + batch_len] = batch["frame_index"].unsqueeze(1).cpu()
        episode_indices[current_idx:current_idx + batch_len] = batch["episode_index"].unsqueeze(1).cpu()
        indices[current_idx:current_idx + batch_len] = batch["index"].unsqueeze(1).cpu()
        task_indices[current_idx:current_idx + batch_len] = batch["task_index"].unsqueeze(1).cpu()

        # Adjust camera key based on config (e.g., for so100_test_3 -> 2 cameras)
        imgs = batch[f"observation.images.{cfg.camera}"]

        if processor:
            if dino_flag:  # DinoV2
                inputs = processor(images=imgs, return_tensors="pt", do_rescale=False)[
                    "pixel_values"
                ].to(device)
            if sig_flag:  # SigLIP
                inputs = processor(
                    text=prompt, images=imgs, padding="max_length", return_tensors="pt"
                ).to(device)
        elif transform:  # Resnet18
            inputs = transform(imgs).to(device)
        else:
            raise ValueError("Either a processor or a transform must be provided.")

        with torch.no_grad():
            if processor:
                if sig_flag:
                    outputs = model(**inputs)
                    outputs = outputs.image_embeds
                elif dino_flag:
                    outputs = model(pixel_values=inputs).last_hidden_state
                    outputs = outputs[:, 1:, :]  # Remove cls token
            else:
                outputs = model(inputs)

        # Pre-allocate features tensor based on model output shape
        if features is None:
            features_shape = [total_frames] + list(outputs.shape[1:])
            # Store features in float16 to save 50% memory (cast back to float32 later)
            features = torch.empty(features_shape, dtype=torch.float16)

        features[current_idx:current_idx + batch_len] = outputs.cpu().half()
        current_idx += batch_len

        # Update progress bar and resource tracking
        cpu_usage = get_cpu_memory_usage()
        gpu_usage = get_gpu_memory_usage()
        pbar.set_description(
            f"{((pbar.n + 1) / total_batches) * 100:5.1f}% | CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}%"
        )
        pbar.update(1)

    # Finished batch loop

    # Truncate if dataloader yielded fewer frames than expected
    if current_idx < total_frames:
        features = features[:current_idx]
        actions = actions[:current_idx]
        timestamps = timestamps[:current_idx]
        frame_indices = frame_indices[:current_idx]
        episode_indices = episode_indices[:current_idx]
        indices = indices[:current_idx]
        task_indices = task_indices[:current_idx]

    data_dict = {
        "features": features,
        "action": actions,
        "timestamp": timestamps,
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": indices,
        "task_index": task_indices,
    }

    pbar.close()
    return data_dict, inputs


def model_extract_features_eval(
    cfg, obs, processor, transform, model, device, dino_flag, sig_flag
):
    """
    Processes images directly as a list of tensors for evaluation mode.

    Args:
        cfg: Configuration object.
        obs: List of observation tensors.
        processor: Preprocessor (e.g., AutoImageProcessor).
        transform: Torchvision transforms.
        model: Feature extraction model.
        device: Target device (GPU or CPU).
        dino_flag (bool): True if using DINOv2.
        sig_flag (bool): True if using SigLIP.

    Returns:
        tuple: (outputs, inputs) containing the extracted features and processed inputs.
    """

    # Prompt for SigLIP
    instructions = cfg.prompt
    prompt = f"What action should the robot take to {instructions}?"

    if processor:
        obs = [o.squeeze() for o in obs]
        if dino_flag:  # DinoV2
            inputs = processor(images=obs, return_tensors="pt", do_rescale=False)[
                "pixel_values"
            ].to(device)
            outputs = model(pixel_values=inputs).last_hidden_state
            outputs = outputs[:, 1:, :]  # Remove cls token

        if sig_flag:  # SigLIP
            inputs = processor(
                text=prompt, images=obs, padding="max_length", return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            outputs = outputs.image_embeds

    elif transform:  # Resnet18
        obs_tensor = torch.stack([img.squeeze() for img in obs]).to(device)
        # Simulate the original to_pil_image -> ToTensor quantization to match features.pt
        obs_tensor = (obs_tensor * 255).to(torch.uint8).to(torch.float32) / 255.0
        inputs = transform(obs_tensor)
        outputs = model(inputs)
    else:
        raise ValueError("Either a processor or a transform must be provided.")

    # Return outputs

    return outputs, inputs


# ============================ DINOv2 ============================


def DinoV2_extract(dataset, device, process, cfg, model_DinoV2, processor_DinoV2):
    """
    Full pipeline to extract features from a Lerobot dataset using a DINOv2 encoder.

    Args:
        dataset: Dataset containing observations.
        device: Target device.
        process (str): 'train' or 'eval'.
        cfg: Configuration object.
        model_DinoV2: DINOv2 model instance.
        processor_DinoV2: DINOv2 processor instance.

    Returns:
        tuple: (data_dict, inputs)
    """
    model_DinoV2.eval()
    transform = None

    if process == "train":
        data_dict, inputs = model_extract_features_train(
            cfg,
            dataset,
            processor_DinoV2,
            transform,
            model_DinoV2,
            device,
            dino_flag=True,
            sig_flag=False,
        )

        save_dir = (
            Path(cfg.encoder_features_dir)
            / f"features_{cfg.encoder}_{cfg.train_dataset_id}"
            / "features.pt"
        )
        save_features(data_dict, save_dir)

    elif process == "eval":
        data_dict, inputs = model_extract_features_eval(
            cfg,
            dataset,
            processor_DinoV2,
            transform,
            model_DinoV2,
            device,
            dino_flag=True,
            sig_flag=False,
        )

    return data_dict, inputs


# ============================ SigLIP ==================================#


def SigLIP_extract(dataset, device, process, cfg, model_siglip, processor_siglip):
    """
    Full pipeline to extract features from a Lerobot dataset using SigLIP.

    Args:
        dataset: Dataset containing observations.
        device: Target device.
        process (str): 'train' or 'eval'.
        cfg: Configuration object.
        model_siglip: SigLIP model instance.
        processor_siglip: SigLIP processor instance.

    Returns:
        tuple: (features, inputs)
    """
    model_siglip.eval()
    transform = None

    if process == "train":
        features, inputs = model_extract_features_train(
            cfg,
            dataset,
            processor_siglip,
            transform,
            model_siglip,
            device,
            dino_flag=False,
            sig_flag=True,
        )

        save_dir = (
            Path(cfg.encoder_features_dir)
            / f"features_{cfg.encoder}_{cfg.train_dataset_id}"
            / "features.pt"
        )
        save_features(features, save_dir)

    elif process == "eval":
        features, inputs = model_extract_features_eval(
            cfg,
            dataset,
            processor_siglip,
            transform,
            model_siglip,
            device,
            dino_flag=False,
            sig_flag=True,
        )

    return features, inputs


# ============================ ResNet18 ============================


def Resnet18_extract(dataset, device, process, cfg, resnet18, transform):
    """
    Full pipeline to extract features from a Lerobot dataset using a ResNet18 encoder.

    Args:
        dataset: Dataset containing observations.
        device: Target device.
        process (str): 'train' or 'eval'.
        cfg: Configuration object.
        resnet18: ResNet18 model instance.
        transform: Torchvision transforms for ResNet18.

    Returns:
        tuple: (features, inputs)
    """
    resnet18.eval()
    processor = None

    if process == "train":
        features, inputs = model_extract_features_train(
            cfg,
            dataset,
            processor,
            transform,
            resnet18,
            device,
            dino_flag=False,
            sig_flag=False,
        )

        save_dir = (
            Path(cfg.encoder_features_dir)
            / f"features_{cfg.encoder}_{cfg.train_dataset_id}"
            / "features.pt"
        )
        save_features(features, save_dir)

    elif process == "eval":
        features, inputs = model_extract_features_eval(
            cfg,
            dataset,
            processor,
            transform,
            resnet18,
            device,
            dino_flag=False,
            sig_flag=False,
        )

    return features, inputs
