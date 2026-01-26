# backend/app/model_utils.py
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Efficient feature extractor using timm (EfficientNet)
def load_backbone(name="tf_efficientnet_lite0"):
    model = timm.create_model(name, pretrained=True, num_classes=0, global_pool='avg')
    model.eval().to(device)
    return model

# CLIP via timm or using a small ViT for embeddings (simple wrapper)
def preprocess_image(pil_image, size=224):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(pil_image).unsqueeze(0).to(device)

def image_to_embedding(model, pil_image, size=224):
    x = preprocess_image(pil_image, size=size)
    with torch.no_grad():
        feats = model(x)  # (1, feat_dim)
    return feats.cpu().numpy().squeeze()
