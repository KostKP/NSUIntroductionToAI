import json
import torch
from torch import nn
import torchvision.models as models
import torchsummary

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
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

    torchsummary.summary(model, input_size=(3, 224, 224))

if __name__ == "__main__":
    main()
