import os
import shutil

def organize_dataset(base_path):
    """
    Separates images into train, val, and test folders based on provided .txt lists.
    """
    images_dir = os.path.join(base_path, "images")
    train_list_path = os.path.join(base_path, "train.txt")
    val_list_path = os.path.join(base_path, "val.txt")
    test_list_path = os.path.join(base_path, "test.txt")

    output_dirs = {
        "train": os.path.join(images_dir, "train"),
        "val": os.path.join(images_dir, "val"),
        "test": os.path.join(images_dir, "test"),
    }

    # Create output directories if they don't exist
    for _, path in output_dirs.items():
        os.makedirs(path, exist_ok=True)

    # Process each list file
    for list_name, list_path in {
        "train": train_list_path,
        "val": val_list_path,
        "test": test_list_path,
    }.items():
        if not os.path.exists(list_path):
            print(f"Warning: {list_path} not found. Skipping {list_name} image separation.")
            continue

        with open(list_path, "r") as f:
            filenames = f.read().splitlines()

        print(f"Processing {list_name} images...")
        for line in filenames:
            # The line in the txt file is like "images\00002475.jpg"
            # We need to extract just the filename "00002475.jpg"
            filename = os.path.basename(line)
            src_path = os.path.join(images_dir, filename)
            dest_path = os.path.join(output_dirs[list_name], filename)

            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dest_path)
                except Exception as e:
                    print(f"Error moving {filename} to {output_dirs[list_name]}: {e}")
            else:
                print(f"Warning: Image {filename} not found at {src_path}. Skipping.")

    print("✅ Images successfully separated into train/val/test folders!")

if __name__ == "__main__":
    dataset_base_path = "c:\\Users\\SHAZ\\Downloads\\face_detection_project\\home\\ubuntu\\face_detection_project_zip\\dataset"
    organize_dataset(dataset_base_path)