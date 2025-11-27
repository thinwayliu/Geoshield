"""
M-Attack: Targeted adversarial attack for image geolocation misdirection.

This module implements targeted adversarial attacks that mislead VLM-based
geolocation models to predict incorrect locations.
"""

import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import hydra
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from config_schema import MainConfig
from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)
from utils import hash_training_config, setup_wandb, ensure_dir

os.environ["WANDB_MODE"] = "disabled"

BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


def get_models(cfg: MainConfig):
    """Initialize feature extraction models based on configuration.

    Args:
        cfg: Configuration object containing model settings.

    Returns:
        Tuple of (feature_extractor, list of models).
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor(models)
    else:
        ensemble_extractor = models[0]

    return ensemble_extractor, models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    """Create ensemble loss function."""
    return EnsembleFeatureLoss(models)


def set_environment(seed: int = 2023):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic: Image.Image) -> torch.Tensor:
    """Convert PIL Image to PyTorch Tensor."""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(Dataset):
    """Dataset that returns images along with their file paths."""

    def __init__(self, root: str, transform=None):
        self.paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0, path


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models_mattack")
def main(cfg: MainConfig):
    """Main entry point for M-Attack."""
    set_environment()
    setup_wandb(cfg, tags=["image_generation"])
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    transform_fn = transforms.Compose([
        transforms.Resize(
            cfg.model.input_res,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Lambda(lambda img: to_tensor(img)),
    ])

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_clean = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    print("Using source crop:", cfg.model.use_source_crop)
    print("Using target crop:", cfg.model.use_target_crop)

    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )

    target_iter = iter(data_loader_target)
    for i, (image_org, _, path_org) in enumerate(data_loader_clean):
        try:
            image_tgt, _, path_tgt = next(target_iter)
        except StopIteration:
            target_iter = iter(data_loader_target)
            image_tgt, _, path_tgt = next(target_iter)

        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            target_crop=target_crop,
        )

    wandb.finish()


def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgt: torch.Tensor,
):
    """Attack a single image pair."""
    image_org = image_org.to(cfg.model.device)
    image_tgt = image_tgt.to(cfg.model.device)

    adv_image = fgsm_attack(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
    )

    config_hash = hash_training_config(cfg) + 'target'

    for path_idx in range(len(path_org)):
        folder, name = (
            os.path.basename(os.path.dirname(path_org[path_idx])),
            os.path.basename(path_org[path_idx]),
        )
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        ext = os.path.splitext(name)[1].lower()
        save_name = name if ext in [".jpeg", ".jpg", ".png"] else name + ".png"
        save_path = os.path.join(folder_to_save, save_name)
        torchvision.utils.save_image(adv_image[path_idx], save_path)


def log_metrics(pbar, metrics: Dict, img_index: int, epoch: int = None):
    """Log metrics to progress bar and wandb."""
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    wandb_metrics = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch
    wandb.log(wandb_metrics)


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
) -> torch.Tensor:
    """Perform FGSM attack to generate adversarial examples.

    Args:
        cfg: Configuration parameters.
        ensemble_extractor: Ensemble feature extractor model.
        ensemble_loss: Ensemble loss function.
        source_crop: Transform for cropping source images.
        target_crop: Transform for cropping target images.
        img_index: Image index for logging.
        image_org: Original source image tensor.
        image_tgt: Target image tensor.

    Returns:
        Adversarial image tensor.
    """
    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc="Attack progress")

    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image


if __name__ == "__main__":
    main()
