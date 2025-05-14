import os
import json
import cv2
import random
import pandas as pd
from pathlib import Path

config = {
    "seed" : 23930,
    "tile_size": 640,
    "tile_overlap": 48,
    "split_ratio" : 0.95,
    "annotations_file" : "./dataset/Heridal/train/_annotations.csv",
    "file_extension" : ".jpg",
    "base_dataset_dir": "./dataset/Heridal/train",
    "cust_dataset_dir": "./dataset/Custom"
}

def extract_dataset_files(path, file_extension):
    content = []
    for item in os.listdir(path):
        item_as_path = os.path.join(path, item)
        if not os.path.isdir(item_as_path) and item_as_path.endswith(file_extension):
            content.append(item_as_path)
    return content

def split_image(image_path, tile_size=768, overlap=64):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    x_tiles = (w - overlap) // (tile_size - overlap) + 1
    y_tiles = (h - overlap) // (tile_size - overlap) + 1
    
    new_w = (tile_size - overlap) * x_tiles + overlap
    new_h = (tile_size - overlap) * y_tiles + overlap
    pad_w = max(new_w - w, 0)
    pad_h = max(new_h - h, 0)
    
    img = cv2.copyMakeBorder(img, pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT)
    
    tiles = []
    coords = []
    
    for y in range(y_tiles):
        for x in range(x_tiles):
            x0 = x*(tile_size - overlap)
            y0 = y*(tile_size - overlap)
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            
            tile = img[y0:y1, x0:x1]
            tiles.append(tile)
            coords.append((x0 - pad_w//2, y0 - pad_h//2, x1 - pad_w//2, y1 - pad_h//2))
    
    return tiles, coords, (pad_w, pad_h)

def process_annotations(image_path, annotations_df, tile_coords, tile_size):
    filename = os.path.basename(image_path)
    img_anns = annotations_df[annotations_df['filename'] == filename]
    
    tile_anns = []
    for idx, (x0, y0, x1, y1) in enumerate(tile_coords):
        tile_ann = []
        for _, ann in img_anns.iterrows():
            axmin = max(ann['xmin'], x0)
            aymin = max(ann['ymin'], y0)
            axmax = min(ann['xmax'], x1)
            aymax = min(ann['ymax'], y1)
            
            if axmin < axmax and aymin < aymax:
                x_center = ((axmin + axmax)/2 - x0) / tile_size
                y_center = ((aymin + aymax)/2 - y0) / tile_size
                width = (axmax - axmin) / tile_size
                height = (aymax - aymin) / tile_size
                
                tile_ann.append(f"0 {x_center} {y_center} {width} {height}")
        
        tile_anns.append(tile_ann)
    
    return tile_anns

def process_dataset(config):
    print("Parsing annotations file...")
    annotations = pd.read_csv(config["annotations_file"])

    print("Parsing dataset directory...")
    dataset_content = extract_dataset_files(config["base_dataset_dir"], config["file_extension"])

    split_index = int(len(dataset_content) * config["split_ratio"])
    random.Random(config["seed"]).shuffle(dataset_content)

    custom_output_dir = Path(config["cust_dataset_dir"])
    custom_output_dir.mkdir(exist_ok=True)

    train_content = dataset_content[:split_index]
    train_output_dir = Path(config["cust_dataset_dir"]) / 'train'
    train_output_dir.mkdir(exist_ok=True)
    Path(config["cust_dataset_dir"] + "/train/images").mkdir(exist_ok=True)
    Path(config["cust_dataset_dir"] + "/train/labels").mkdir(exist_ok=True)

    val_content = dataset_content[split_index:]
    val_output_dir = Path(config["cust_dataset_dir"]) / 'val'
    val_output_dir.mkdir(exist_ok=True)
    Path(config["cust_dataset_dir"] + "/val/images").mkdir(exist_ok=True)
    Path(config["cust_dataset_dir"] + "/val/labels").mkdir(exist_ok=True)

    # Train part
    for img_path in train_content:
        tiles, coords, _ = split_image(img_path, config['tile_size'], config['tile_overlap'])
        tile_anns = process_annotations(img_path, annotations, coords, config['tile_size'])
        
        base = os.path.basename(img_path).split('.')[0]
        for i, (tile, ann) in enumerate(zip(tiles, tile_anns)):
            if not ann:
                continue
            tile_name = f"{base}_tile_{i}.jpg"
            cv2.imwrite(str(train_output_dir / "images" / tile_name), tile)
            
            label_file = train_output_dir / "labels" / f"{tile_name.replace('.jpg', '.txt')}"
            with open(label_file, 'w') as f:
                f.write('\n'.join(ann))

    # Val part
    for img_path in val_content:
        tiles, coords, _ = split_image(img_path, config['tile_size'], config['tile_overlap'])
        tile_anns = process_annotations(img_path, annotations, coords, config['tile_size'])
        
        base = os.path.basename(img_path).split('.')[0]
        for i, (tile, ann) in enumerate(zip(tiles, tile_anns)):
            tile_name = f"{base}_tile_{i}.jpg"
            cv2.imwrite(str(val_output_dir / "images" / tile_name), tile)
            
            label_file = val_output_dir / "labels" / f"{tile_name.replace('.jpg', '.txt')}"
            with open(label_file, 'w') as f:
                f.write('\n'.join(ann))

    print("All files saved...")

if __name__ == "__main__":
    process_dataset(config)
