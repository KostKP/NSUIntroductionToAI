import os
import json
import time
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadsDataset(Dataset):
    def __init__(self, metadata_df, dataset_dir, split, transform=None):
        self.metadata = metadata_df[metadata_df['split'] == split]
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.classes = pd.read_csv(os.path.join(dataset_dir, 'label_class_dict.csv'))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx]['tiff_image_path'])
        label_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx]['tif_label_path'])

        image = np.array(Image.open(img_path).convert('RGB'))
        label_img = np.array(Image.open(label_path).convert('RGB'))

        label = np.all(label_img == [255, 255, 255], axis=-1).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        return image, label.unsqueeze(0)  # (1, H, W)

base_transforms = [
    A.Resize(512, 512),
]

train_transform = A.Compose(base_transforms + [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
] + [ToTensorV2()])

transform = A.Compose(base_transforms + [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
] + [ToTensorV2()])

def build_model():
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model

def plot_loss(train_losses, val_losses):
    sns.set(style="ticks")
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o', color='b')
    for epoch, loss in zip(epochs, train_losses):
        plt.text(epoch, loss, f"{loss:.4f}", ha='right', va='bottom', fontsize=10)
    plt.plot(epochs, val_losses, label="Validation Loss", marker='o', color='r')
    for epoch, loss in zip(epochs, val_losses):
        plt.text(epoch, loss, f"{loss:.4f}", ha='left', va='bottom', fontsize=10)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf8") as f:
        return json.load(f)

def main():
    config = load_config()
    global_config = config['Global']
    train_config = config['Train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model().to(device)

    metadata = pd.read_csv(train_config['metadata'])

    model_pth = global_config["model_pth"]

    if os.path.exists(model_pth):
        print(f"Loading existing model from {model_pth}")
        model.load_state_dict(torch.load(model_pth, map_location=device, weights_only=True))
    else:
        print("Starting training...")

        train_dataset = RoadsDataset(metadata, global_config['dataset_dir'], 'train', train_transform)
        val_dataset = RoadsDataset(metadata, global_config['dataset_dir'], 'val', transform)

        train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()

        train_losses = []
        val_losses = []

        for epoch in range(train_config['epochs']):
            start_time = time.time()

            # Training stage

            model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
            
            epoch_loss /= len(train_dataset)
            train_losses.append(epoch_loss)

            # Validation stage

            model.eval()
            val_loss = 0.0
            val_iou = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs) > 0.5
                    intersection = (preds & labels.bool()).sum()
                    union = (preds | labels.bool()).sum()
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    val_iou += iou.item() * images.size(0)

            val_loss = val_loss / len(val_dataset)
            val_iou = val_iou / len(val_dataset)

            val_losses.append(val_loss)

            # Epoch result

            epoch_time = time.time() - start_time
            print(f"  Epoch {str(epoch + 1).zfill(len(str(train_config['epochs'])))}/{train_config['epochs']}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Time: {epoch_time:.2f} sec.")

            if train_config.get('save_every_epoch', False):
                epoch_pth = f"{model_pth[:-4]}_{epoch+1:03d}.pth"
                torch.save(model.state_dict(), epoch_pth)

        torch.save(model.state_dict(), model_pth)
        print(f"Model saved as {model_pth}")
        plot_loss(train_losses, val_losses)

    print("Starting testing...")

    test_dataset = RoadsDataset(metadata, global_config['dataset_dir'], 'test', transform)

    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4)
    
    model.eval()

    total_iou = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.bool()
            labels_bool = labels.bool()

            intersection = (preds & labels_bool).sum(dim=(1, 2, 3)).float()
            union = (preds | labels_bool).sum(dim=(1, 2, 3)).float()
            iou = (intersection + 1e-6) / (union + 1e-6)

            # Dice Coefficient
            dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + labels_bool.sum(dim=(1, 2, 3)) + 1e-6)

            # Pixel Accuracy
            correct = (preds == labels_bool).sum(dim=(1, 2, 3)).float()
            total = torch.tensor(labels_bool.shape[1] * labels_bool.shape[2] * labels_bool.shape[3]).float()
            pixel_acc = correct / total

            # Precision and Recall
            tp = intersection
            fp = (preds & ~labels_bool).sum(dim=(1, 2, 3)).float()
            fn = (~preds & labels_bool).sum(dim=(1, 2, 3)).float()

            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)

            batch_size = images.size(0)
            num_samples += batch_size
            total_iou += iou.sum().item()
            total_dice += dice.sum().item()
            total_pixel_acc += pixel_acc.sum().item()
            total_precision += precision.sum().item()
            total_recall += recall.sum().item()

    test_iou = total_iou / num_samples
    test_dice = total_dice / num_samples
    test_pixel_acc = total_pixel_acc / num_samples
    test_precision = total_precision / num_samples
    test_recall = total_recall / num_samples

    print(f"Test Results:")
    print(f"  IoU: {test_iou:.8f}")
    print(f"  Dice Coefficient: {test_dice:.8f}")
    print(f"  Pixel Accuracy: {test_pixel_acc:.8f}")
    print(f"  Precision: {test_precision:.8f}")
    print(f"  Recall: {test_recall:.8f}")

    # Example

    indices = random.sample(range(len(test_dataset)), 3)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for row, idx in enumerate(indices):
        image, true_mask = test_dataset[idx]
        
        with torch.no_grad():
            pred = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(float)
        
        true_mask_np = true_mask.squeeze().numpy()
        diff_mask = np.zeros((*true_mask_np.shape, 3), dtype=np.uint8) + 255
        
        # Green (True Positive)
        tp_mask = (true_mask_np == 1) & (pred_mask == 1)
        diff_mask[tp_mask] = [0, 255, 0]
        
        # Blue (False Negative)
        fn_mask = (true_mask_np == 1) & (pred_mask == 0)
        diff_mask[fn_mask] = [0, 0, 255]
        
        # Red (False Positive)
        fp_mask = (true_mask_np == 0) & (pred_mask == 1)
        diff_mask[fp_mask] = [255, 0, 0]

        img = image.numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)

        titles = ['Original Image', 'True Mask', 'Prediction', 'Comparison Mask']
        data_to_show = [
            img,
            true_mask.squeeze().numpy(),
            pred_mask,
            diff_mask
        ]

        diff_mask = np.copy(diff_mask)
        h, w, _ = diff_mask.shape
        diff_mask[0, :, :] = [0, 0, 0]
        diff_mask[-1, :, :] = [0, 0, 0]
        diff_mask[:, 0, :] = [0, 0, 0]
        diff_mask[:, -1, :] = [0, 0, 0]
        
        for col in range(4):
            ax = axes[row, col]
            data = data_to_show[col]
            
            if col == 3:
                ax.imshow(diff_mask)
            else:
                cmap = 'gray' if col in [1,2] else None
                ax.imshow(data, cmap=cmap)
            
            ax.set_title(titles[col], fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
