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
import pandas as pd
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

class RoadsDataset(Dataset):
    def __init__(self, metadata_df, dataset_dir, split, transform=None):
        self.metadata = metadata_df[metadata_df['split'] == split]
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.classes = pd.read_csv(os.path.join(dataset_dir, 'label_class_dict.csv'))
        self.split = split

    def __len__(self):
        return len(self.metadata) * 9

    def _get_tile(self, image, label, tile_idx):
        i = tile_idx // 3
        j = tile_idx % 3
        x_start = i * 500
        y_start = j * 500
        image_tile = image[x_start:x_start+500, y_start:y_start+500, :]
        label_tile = label[x_start:x_start+500, y_start:y_start+500]

        image_tile = cv2.copyMakeBorder(image_tile, 6,6,6,6, cv2.BORDER_REFLECT)
        label_tile = cv2.copyMakeBorder(label_tile.astype(np.float32), 6,6,6,6, cv2.BORDER_CONSTANT, value=0)
        return image_tile, label_tile

    def __getitem__(self, idx):
        original_idx = idx // 9
        tile_idx = idx % 9

        img_path = os.path.join(self.dataset_dir, self.metadata.iloc[original_idx]['tiff_image_path'])
        label_path = os.path.join(self.dataset_dir, self.metadata.iloc[original_idx]['tif_label_path'])

        image = np.array(Image.open(img_path).convert('RGB'))
        label_img = np.array(Image.open(label_path).convert('RGB'))
        label = np.all(label_img == [255, 255, 255], axis=-1).astype(np.float32)

        image_tile, label_tile = self._get_tile(image, label, tile_idx)

        if self.transform:
            transformed = self.transform(image=image_tile, mask=label_tile)
            image_transformed = transformed['image']
            label_transformed = transformed['mask']
        else:
            image_transformed = ToTensorV2()(image=image_tile, mask=label_tile)['image']
            label_transformed = ToTensorV2()(image=image_tile, mask=label_tile)['mask']

        if self.split == 'test':
            i = tile_idx // 3
            j = tile_idx % 3
            return image_transformed, label_transformed.unsqueeze(0), original_idx, i, j
        else:
            return image_transformed, label_transformed.unsqueeze(0)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=mean, std=std)
] + [ToTensorV2()])

val_transform = A.Compose([
    A.Normalize(mean=mean, std=std)
] + [ToTensorV2()])

transform = A.Compose([
    A.Normalize(mean=mean, std=std)
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
        val_dataset = RoadsDataset(metadata, global_config['dataset_dir'], 'val', val_transform)

        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=8)

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
                
                # Cut padding
                outputs_cropped = outputs[:, :, 6:-6, 6:-6]
                labels_cropped = labels[:, :, 6:-6, 6:-6]
                
                loss = criterion(outputs_cropped, labels_cropped)
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
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    
                    # Cut padding
                    outputs_cropped = outputs[:, :, 6:-6, 6:-6]
                    labels_cropped = labels[:, :, 6:-6, 6:-6]
                    
                    loss = criterion(outputs_cropped, labels_cropped)
                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs_cropped) > 0.5
                    intersection = (preds & labels_cropped.bool()).sum()
                    union = (preds | labels_cropped.bool()).sum()
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
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=8)
    
    model.eval()

    r_tile_data = {'map_image': [], 'true_mask': [], 'pred_mask': [], 'orig_idx': [], 'i': [], 'j': []}
    output_data = {'map_image': [], 'true_mask': [], 'pred_mask': []}
    metrics = {'iou': [], 'dice': [], 'pixel_acc': [], 'precision': [], 'recall': []}

    with torch.no_grad():
        for images, labels, orig_indices, i_coords, j_coords in test_loader:
            images_np = images.numpy()
            images = images.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            labels = labels.squeeze(1).cpu().numpy().astype(np.uint8)

            preds = preds[:, 6:-6, 6:-6]
            labels = labels[:, 6:-6, 6:-6]
            images_np = images_np[:, :, 6:-6, 6:-6]  # (B, C, H, W)

            for k in range(len(orig_indices)):
                orig_idx = orig_indices[k].item()
                i = i_coords[k].item()
                j = j_coords[k].item()

                r_tile_data['map_image'].append(images_np[k].transpose(1, 2, 0))  # (H, W, C)
                r_tile_data['true_mask'].append(labels[k])
                r_tile_data['pred_mask'].append(preds[k])
                r_tile_data['orig_idx'].append(orig_idx)
                r_tile_data['i'].append(i)
                r_tile_data['j'].append(j)

    tile_df = pd.DataFrame(r_tile_data)
    
    for orig_idx, group in tile_df.groupby('orig_idx'):
        sorted_tiles = group.sort_values(by=['i', 'j'])
        
        pred_full = np.zeros((1500, 1500), dtype=np.uint8)
        label_full = np.zeros((1500, 1500), dtype=np.uint8)
        img_full = np.zeros((1500, 1500, 3), dtype=np.uint8)
        
        for _, row in sorted_tiles.iterrows():
            i = row['i']
            j = row['j']
            x_start = i * 500
            y_start = j * 500
            
            pred_tile = row['pred_mask']
            label_tile = row['true_mask']
            img_tile = (row['map_image'] * np.array(std) + np.array(mean)) * 255
            img_tile = np.clip(img_tile, 0, 255).astype(np.uint8)
            
            pred_full[x_start:x_start+500, y_start:y_start+500] = pred_tile
            label_full[x_start:x_start+500, y_start:y_start+500] = label_tile
            img_full[x_start:x_start+500, y_start:y_start+500, :] = img_tile

        output_data['map_image'].append(img_full)
        output_data['true_mask'].append(label_full.astype(bool))
        output_data['pred_mask'].append(pred_full.astype(bool))

        pred = pred_full.astype(bool)
        true = label_full.astype(bool)
        
        intersection = np.logical_and(pred, true).sum()
        union = np.logical_or(pred, true).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        dice = (2 * intersection + 1e-6) / (pred.sum() + true.sum() + 1e-6)
        pixel_acc = np.equal(pred, true).sum() / pred.size
        tp = intersection
        fp = np.logical_and(pred, ~true).sum()
        fn = np.logical_and(~pred, true).sum()
        
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        
        metrics['iou'].append(iou)
        metrics['dice'].append(dice)
        metrics['pixel_acc'].append(pixel_acc)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

    print(f"Test Results:")
    print(f"  IoU: {np.mean(metrics['iou']):.8f}")
    print(f"  Dice Coefficient: {np.mean(metrics['dice']):.8f}")
    print(f"  Pixel Accuracy: {np.mean(metrics['pixel_acc']):.8f}")
    print(f"  Precision: {np.mean(metrics['precision']):.8f}")
    print(f"  Recall: {np.mean(metrics['recall']):.8f}")

    # Example

    indices = random.sample(range(len(output_data['map_image'])), 3)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for row, idx in enumerate(indices):
        true_mask = output_data['true_mask'][idx]
        pred_mask = output_data['pred_mask'][idx]

        diff_mask = np.zeros((*true_mask.shape, 3), dtype=np.uint8) + 255
        
        # Green (True Positive)
        tp_mask = (true_mask == 1) & (pred_mask == 1)
        diff_mask[tp_mask] = [0, 255, 0]
        
        # Blue (False Negative)
        fn_mask = (true_mask == 1) & (pred_mask == 0)
        diff_mask[fn_mask] = [0, 0, 255]
        
        # Red (False Positive)
        fp_mask = (true_mask == 0) & (pred_mask == 1)
        diff_mask[fp_mask] = [255, 0, 0]


        titles = ['Original Image', 'True Mask', 'Prediction', 'Comparison Mask']
        data_to_show = [
            output_data['map_image'][idx],
            true_mask,
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
