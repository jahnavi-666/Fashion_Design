# backend/app/clip_utils.py
import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Device: will be "cuda" if available, else "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Load once at import time (may download weights the first time)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
model.eval()

def image_to_clip_embedding(pil_image, normalize=True):
    """
    Input: PIL.Image
    Output: numpy float32 vector (512), normalized if requested
    """
    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)  # (1, dim)
    emb = emb.cpu().numpy().reshape(-1).astype("float32")
    if normalize:
        norm = np.linalg.norm(emb) + 1e-12
        emb = emb / norm
    return emb
