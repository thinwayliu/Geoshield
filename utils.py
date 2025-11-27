"""Shared utilities for adversarial attack."""

import os
import json
import hashlib
import base64
from typing import Dict
from omegaconf import OmegaConf
import wandb
from config_schema import MainConfig


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.

    Args:
        cfg: Configuration object containing model settings.

    Returns:
        MD5 hash string of the config parameters.
    """
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)

    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }

    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.

    Args:
        cfg: Configuration object containing wandb settings.
        tags: Optional list of tags for the run.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=tags,
    )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to image file.

    Returns:
        Base64 encoded image string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.

    Args:
        cfg: Configuration object.
        config_hash: Hash of training config.

    Returns:
        Dictionary containing output paths.
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    }
