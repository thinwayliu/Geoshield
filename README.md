# GeoShield

GeoShield is a research tool for generating adversarial perturbations that protect image geolocation privacy. It creates imperceptible perturbations that prevent Vision-Language Models (VLMs) from accurately predicting image geolocation while preserving semantic content.

## Overview

GeoShield implements two complementary attack strategies:

- **GeoShield**: An untargeted attack that disrupts geolocation prediction by introducing geo-semantic aware perturbations
- **M-Attack**: A targeted attack that misleads models to predict specific incorrect locations

Both methods leverage ensemble CLIP models as surrogate models and employ FGSM-based iterative optimization for generating robust adversarial examples.

## Features

- Multi-model ensemble approach for improved transferability
- Geo-semantic aware loss function
- Region-based attack with GroundingDINO integration
- Configurable perturbation budgets and optimization parameters
- Support for multiple CLIP backbone variants

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Basic Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/GeoShield.git
cd GeoShield
```

2. Install required dependencies:
```bash
pip install torch torchvision transformers hydra-core omegaconf wandb tqdm pillow numpy
```

### GroundingDINO Setup (Optional)

For region-aware attacks using object detection, you need to set up GroundingDINO:

1. Clone the GroundingDINO repository:
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
```

2. Install GroundingDINO:
```bash
pip install -e .
```

3. Download pretrained weights:
```bash
mkdir weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

4. Run object detection on your images:
```bash
python demo/inference_on_a_image.py \
    -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
    -p weights/groundingdino_swint_ogc.pth \
    -i <path_to_your_image> \
    -o detections.json \
    -t "building, landmark, sign, architecture, structure"
```

5. Use the generated JSON file with GeoShield:
```bash
python geoshield.py \
    data.cle_data_path=<clean_images_path> \
    data.tgt_data_path=<target_images_path> \
    data.bbox_json_path=detections.json
```

## Usage

### GeoShield (Untargeted Attack)

Generate adversarial examples that disrupt geolocation prediction:

```bash
python geoshield.py \
    data.cle_data_path=data/clean_images \
    data.tgt_data_path=data/target_images \
    data.output=./output \
    data.num_samples=100 \
    optim.epsilon=8 \
    optim.steps=100
```

### M-Attack (Targeted Attack)

Generate adversarial examples that mislead to a specific target location:

```bash
python m-attack.py \
    data.cle_data_path=data/clean_images \
    data.tgt_data_path=data/target_images \
    data.output=./output \
    data.num_samples=100 \
    optim.epsilon=8 \
    optim.steps=100
```

### Configuration Parameters

All configuration is managed through Hydra. Key parameters include:

#### Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.batch_size` | Batch size for processing | 1 |
| `data.num_samples` | Number of images to process | 100 |
| `data.cle_data_path` | Path to clean/source images | "data/clean_images" |
| `data.tgt_data_path` | Path to target images | "data/target_images" |
| `data.output` | Output directory | "./output" |
| `data.bbox_json_path` | Path to GroundingDINO detection results | "" |

#### Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `optim.epsilon` | Maximum L∞ perturbation magnitude (0-255) | 8 |
| `optim.alpha` | Step size for FGSM iterations | 1.0 |
| `optim.steps` | Number of optimization iterations | 100 |

#### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.input_res` | Input image resolution | 640 |
| `model.ensemble` | Use ensemble of models | true |
| `model.backbone` | List of CLIP models to use | ["B16", "B32", "Laion"] |
| `model.device` | Computation device | "cuda:0" |
| `model.use_source_crop` | Apply random crop to source | true |
| `model.use_target_crop` | Apply random crop to target | true |
| `model.crop_scale` | Random crop scale range | [0.5, 0.9] |

### Available CLIP Backbones

- `B16`: CLIP ViT-B/16 (openai/clip-vit-base-patch16)
- `B32`: CLIP ViT-B/32 (openai/clip-vit-base-patch32)
- `L336`: CLIP ViT-L/14-336 (openai/clip-vit-large-patch14-336)
- `Laion`: LAION CLIP ViT-G/14 (laion/CLIP-ViT-G-14-laion2B-s12B-b42K)

### VLM Integration for Image Description

GeoShield uses VLM-based image descriptions for geo-semantic loss. To integrate your VLM API:

Edit the `describe_image_placeholder` function in `geoshield.py`:

```python
def describe_image_placeholder(image_path: str) -> str:
    # Example: OpenAI GPT-4V
    # from openai import OpenAI
    # client = OpenAI(api_key="your-api-key")
    # response = client.chat.completions.create(
    #     model="gpt-4-vision-preview",
    #     messages=[{
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Describe this image focusing on objects and scene, not location."},
    #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    #         ]
    #     }]
    # )
    # return response.choices[0].message.content

    return "Your VLM description here"
```

## Project Structure

```
GeoShield/
├── geoshield.py                           # GeoShield main script
├── m-attack.py                            # M-Attack script
├── config_schema.py                       # Configuration schemas
├── utils.py                               # Utility functions
├── config/
│   ├── ensemble_3models.yaml             # GeoShield config
│   └── ensemble_3models_mattack.yaml     # M-Attack config
└── surrogates/
    ├── __init__.py
    └── FeatureExtractors/
        ├── __init__.py
        ├── Base.py                        # Base classes and ensemble loss
        ├── ClipB16.py                     # CLIP ViT-B/16 extractor
        ├── ClipB32.py                     # CLIP ViT-B/32 extractor
        ├── ClipL336.py                    # CLIP ViT-L/14-336 extractor
        └── ClipLaion.py                   # LAION CLIP extractor
```

## Examples

### Basic Usage

```bash
# Run GeoShield with default settings
python geoshield.py

# Run M-Attack with custom parameters
python m-attack.py \
    optim.epsilon=16 \
    optim.steps=200 \
    model.backbone=["B16","B32","L336"]
```

### Advanced Configuration

Create a custom config file `config/custom.yaml`:

```yaml
data:
  batch_size: 1
  num_samples: 50
  cle_data_path: "data/my_images"
  tgt_data_path: "data/target_locations"
  output: "./results"
  bbox_json_path: "detections.json"

optim:
  alpha: 1.0
  epsilon: 8
  steps: 150

model:
  input_res: 640
  use_source_crop: true
  use_target_crop: true
  crop_scale: [0.5, 0.9]
  ensemble: true
  device: "cuda:0"
  backbone: ["B16", "B32", "Laion"]

wandb:
  project: "my-geoshield-project"
  entity: ""

attack: 'fgsm'
```

Run with custom config:
```bash
python geoshield.py --config-name=custom
```

## Technical Details

### Attack Algorithm

GeoShield employs a iterative FGSM approach with the following loss components:

1. **Feature Matching Loss**: Maximizes cosine similarity between adversarial image features and target image features
2. **Geo-Semantic Loss**: Ensures perturbations move features away from geolocation-indicative directions while preserving non-geo semantic content
3. **Region-Aware Sampling**: Probabilistically samples object regions for focused perturbations

The optimization objective:

```
min L = -sim(f(x_adv), f(x_target)) - geo_loss(f(x_adv), text, f(x_target))
```

where `f(·)` represents the ensemble feature extractor.

### Perturbation Constraints

- L∞ norm constraint: `||δ||∞ ≤ ε`
- Pixel value range: `[0, 255]`
- Default ε = 8 (approximately 3% of pixel range)

## Limitations

- Perturbations are optimized for CLIP-based models and may not transfer perfectly to all VLMs
- Computational cost scales with number of ensemble models and optimization steps
- Requires paired source and target images for M-Attack

## Research Use Only

This tool is provided for research purposes only. Users are responsible for ensuring compliance with applicable laws and regulations regarding adversarial examples and privacy protection.

## Acknowledgements

This project builds upon and is inspired by:

- **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**: Open-set object detection with grounding DINO
- **[M-Attack](https://github.com/VILA-Lab/M-Attack)**: Adversarial attacks on vision-language models

We thank the authors of these works for their contributions to the research community.

## Citation

If you use GeoShield in your research, please cite:

```bibtex
@misc{geoshield2024,
  title={GeoShield: Adversarial Perturbation for Geolocation Privacy Protection},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-username/GeoShield}}
}
```

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact [liuxinwei@iie.ac.cn].
