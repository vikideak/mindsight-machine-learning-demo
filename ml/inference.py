import torch
from torchvision import transforms, models
from PIL import Image
import json

def load_model():
    """Loads model and label mapping."""
    with open("ml/labels.json") as f:
        label_map = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load("ml/model.pth", map_location="cpu"))
    model.eval()
    return model, label_map

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    """Predicts the class of an image file path."""
    model, label_map = load_model()
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return label_map[str(pred.item())], float(confidence.item())
