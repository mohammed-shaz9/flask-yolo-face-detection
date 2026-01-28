import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths
DATASET_DIR = 'dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images') # Assuming images are in dataset/images
LABELS_DIR = os.path.join(DATASET_DIR, 'labels') # YOLO format labels

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# Load the CSV file
csv_path = os.path.join(DATASET_DIR, 'faces.csv')
df = pd.read_csv(csv_path)

def convert_to_yolo(x0, y0, x1, y1, img_width, img_height):
    # Calculate center x, center y, width, and height
    center_x = (x0 + x1) / 2 / img_width
    center_y = (y0 + y1) / 2 / img_height
    bbox_width = (x1 - x0) / img_width
    bbox_height = (y1 - y0) / img_height
    return center_x, center_y, bbox_width, bbox_height

# Group by image name to process all bounding boxes for an image at once
for img_name, group in tqdm(df.groupby('image_name'), desc="Processing images"):
    label_file_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt'))
    with open(label_file_path, 'w') as f:
        for index, row in group.iterrows():
            img_width = row['width']
            img_height = row['height']
            x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']

            # Convert to YOLO format
            center_x, center_y, bbox_width, bbox_height = convert_to_yolo(x0, y0, x1, y1, img_width, img_height)

            # Write to file (class_id is 0 for 'face')
            f.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print("YOLO format annotations created.")

# Split data into train, validation, and test sets
image_files = df['image_name'].unique()

train_val_files, test_files = train_test_split(image_files, test_size=0.15, random_state=42)
train_files, val_files = train_test_split(train_val_files, test_size=0.176, random_state=42) # 0.176 of 0.85 is approx 0.15 of total

# Create train.txt, val.txt, test.txt for YOLO
def write_image_list(file_list, output_path, base_dir):
    with open(output_path, 'w') as f:
        for img_name in file_list:
            f.write(os.path.join(base_dir, img_name) + '\n')

# Assuming images will be moved to dataset/images/train, dataset/images/val, dataset/images/test
# For now, just create the lists based on the original image paths
# The actual moving of images will be a manual step or handled by a separate script/tool

write_image_list(train_files, os.path.join(DATASET_DIR, 'train.txt'), 'images')
write_image_list(val_files, os.path.join(DATASET_DIR, 'val.txt'), 'images')
write_image_list(test_files, os.path.join(DATASET_DIR, 'test.txt'), 'images')

print("Dataset split into train, validation, and test lists.")
print("Please ensure your image files are organized into 'dataset/images' directory.")
print("You will also need to manually move images into 'dataset/images/train', 'dataset/images/val', and 'dataset/images/test' based on the generated .txt files.")