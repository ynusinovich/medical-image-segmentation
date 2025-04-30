import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "models/best_model.pth"
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=3,
    ignore_mismatched_sizes=True,
    local_files_only=True,
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image size (should match your training input size, e.g., 256)
image_size = 256


def preprocess(image_bytes):
    """Load image bytes and preprocess."""
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize((image_size, image_size))
    image = np.array(image).transpose(2, 0, 1)  # (3, H, W)
    image = torch.from_numpy(image).unsqueeze(0).float() / 255.0  # (1, 3, H, W)
    return image


def predict(image_tensor):
    """Run inference and return predicted mask as numpy array."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(pixel_values=image_tensor).logits  # (1, C, h, w)
        outputs = F.interpolate(
            outputs, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        preds = torch.argmax(outputs, dim=1)  # (1, H, W)
    return preds.squeeze(0).cpu().numpy()  # (H, W)
