"""CLIP ViT-B/32 feature extractor."""

import torch
from transformers import CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor
from torchvision import transforms


class ClipB32FeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using CLIP ViT-B/32 model."""

    def __init__(self):
        super(ClipB32FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def forward(self, x):
        """Extract image features."""
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def extract_text_features(self, texts, device):
        """Extract text features."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features
