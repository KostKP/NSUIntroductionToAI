import os
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class DictImageDataset(Dataset):
    def __init__(self, dataset_map, class_to_idx, transform=None):
        self.transform = transform
        self.data = []
        for label, file_paths in dataset_map.items():
            for path in file_paths:
                self.data.append((path, label))
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf8") as f:
        return json.load(f)

def plot_loss(losses):
    sns.set(style="ticks")
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(losses) + 1))
    plt.plot(epochs, losses, label="Training Loss", marker='o')
    for epoch, loss in zip(epochs, losses):
        plt.text(epoch, loss, f"{loss:.4f}", ha='right', va='bottom', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_confusion_matrix(class_names, all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False, annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Normalized Confusion Matrix")
    plt.show()

def main():
    config = load_config()
    global_config = config["Global"]
    train_config = config["Train"]
    test_config = config["Test"]

    model_pth = global_config["model_pth"]
    train_map_path = train_config["dataset_map"]
    test_map_path = test_config["dataset_map"]

    with open(global_config["dataset_map"], "r", encoding="utf8") as f:
        global_map = json.load(f)
    class_names = list(global_map.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    with open(test_map_path, "r", encoding="utf8") as f:
        test_map = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_names))
    )
    model = model.to(device)

    if os.path.exists(model_pth):
        print(f"Loading existing model from {model_pth}")
        model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
    else:
        with open(train_map_path, "r", encoding="utf8") as f:
            train_map = json.load(f)

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = DictImageDataset(train_map, class_to_idx, train_transform)
        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=16)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])

        train_losses = []
        print("Starting training...")
        for epoch in range(train_config["epochs"]):
            start_time = time.time()
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_dataset)
            train_losses.append(epoch_loss)
            epoch_time = time.time() - start_time

            print(f"  Epoch {str(epoch + 1).zfill(len(str(train_config['epochs'])))}/{train_config['epochs']}, Loss: {epoch_loss:.8f}, Time: {epoch_time:.2f} sec.")

            if train_config["save_every_epoch"]:
                epoch_pth = f"{model_pth}_{epoch+1:03d}"
                torch.save(model.state_dict(), epoch_pth)
        
        print(f"Model saved as {model_pth}")
        torch.save(model.state_dict(), model_pth)
        
        plot_loss(train_losses)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = DictImageDataset(test_map, class_to_idx, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=16)

    model.eval()
    all_preds = []
    all_labels = []
    print("Starting testing...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

    plot_confusion_matrix(class_names, all_labels, all_preds)

if __name__ == "__main__":
    main()
