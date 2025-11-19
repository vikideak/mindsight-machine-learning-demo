import torch
from torchvision import models
import json
from ml.transforms import get_transform

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def load_model():
    with open("ml/labels.json") as f:
        label_map = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load("ml/model.pth", map_location="cpu"))
    model.eval()
    return model, label_map

transform = get_transform()

def predict_image(image, model, label_map):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return label_map[str(pred.item())], float(confidence.item())

