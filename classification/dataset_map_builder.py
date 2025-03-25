import json
import os

def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf8") as f:
        return json.load(f)

def extract_dataset(path, file_extension):
    dataset_content = {}

    for entry in os.listdir(path):
        entry_as_path = os.path.join(path, entry)
        if os.path.isdir(entry_as_path):
            class_content = []
            for item in os.listdir(entry_as_path):
                item_as_path = os.path.join(entry_as_path, item)
                if not os.path.isdir(item_as_path) and item_as_path.endswith(file_extension):
                    class_content.append(item_as_path)
            if len(class_content) > 0:
                dataset_content[entry] = class_content
        else:
            continue  # Skip file
    return dataset_content

def main():
    config = load_config()
    global_config = config["Global"]
    
    dataset_dir = global_config["dataset_dir"]
    map_file_path = global_config["dataset_map"]
    file_extension = global_config["file_extension"]

    print("Parsing dataset directory...")
    dataset_content = extract_dataset(dataset_dir, file_extension)

    print("Found " + str(len(dataset_content)) + " classes:")
    for class_name in dataset_content.keys():
        print("  " + class_name + " - " + str(len(dataset_content[class_name])) + 
              (" files" if len(dataset_content[class_name]) > 1 else " file"))

    with open(map_file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(dataset_content, ensure_ascii=False, indent=4))

    print("Map saved as " + map_file_path)

if __name__ == "__main__":
    main()
