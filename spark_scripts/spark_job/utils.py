import json
from PIL import Image
import torch
import torchvision.transforms as T

def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])
    return transform(image)

def load_tile_metadata(json_path = 'spark_scripts/data/tile_metadata.json'):
    with open(json_path) as f:
        return json.load(f)


