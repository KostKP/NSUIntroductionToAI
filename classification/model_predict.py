import argparse
import json
import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Predict image class.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"The file {args.image_path} does not exist.")

    config = load_config()
    global_config = config["Global"]

    with open(global_config["dataset_map"], "r", encoding="utf-8") as f:
        global_map = json.load(f)
    class_names = list(global_map.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_names))
    )
    model.load_state_dict(torch.load(global_config["model_pth"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

    print(predicted_class)

if __name__ == "__main__":
    main()
