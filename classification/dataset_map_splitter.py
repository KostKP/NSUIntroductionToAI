import json
import random

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf8") as f:
        return json.load(f)

def main():
    config = load_config()
    global_config = config["Global"]
    train_config = config["Train"]
    test_config = config["Test"]
    
    dataset_map_all = global_config["dataset_map"]
    dataset_map_trn = train_config["dataset_map"]
    dataset_map_tst = test_config["dataset_map"]

    split_ratio = global_config["dataset_split_ratio"]

    with open(dataset_map_all, "r", encoding="utf8") as f:
        dataset_content = json.load(f)

    train_set = {}
    test_set = {}

    for class_name, images in dataset_content.items():
        random.shuffle(images) 
        split_index = int(len(images) * split_ratio)
        
        train_set[class_name] = images[:split_index]
        test_set[class_name] = images[split_index:]

    with open(dataset_map_trn, "w", encoding="utf8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)

    with open(dataset_map_tst, "w", encoding="utf8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=4)

    print(f"Data splited for train ({dataset_map_trn}) and test ({dataset_map_tst})")

if __name__ == "__main__":
    main()
