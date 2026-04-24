# src/infer.py
import torch
from torchvision import transforms
from PIL import Image
from src.medical_model import MedicalNet
import pathlib

MODELS_DIR = pathlib.Path("models")
BEST_PATH = MODELS_DIR / "best_model.pth"

# load model once
_device = torch.device("cpu")
_model = MedicalNet(num_classes=2)
state = torch.load(BEST_PATH, map_location=_device)
_model.load_state_dict(state)
_model.eval()

# preprocessing
_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image_path: str):
    """Run inference on a single image path and return probabilities."""
    img = Image.open(image_path).convert("L").convert("RGB")
    x = _tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).tolist()[0]

    return {"NORMAL": float(probs[0]), "PNEUMONIA": float(probs[1])}
