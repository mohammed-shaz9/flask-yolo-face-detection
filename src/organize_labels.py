import os
import shutil

def organize_labels(dataset_path):
    labels_path = os.path.join(dataset_path, 'labels')
    train_labels_path = os.path.join(labels_path, 'train')
    val_labels_path = os.path.join(labels_path, 'val')
    test_labels_path = os.path.join(labels_path, 'test')

    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    for split, dest_path in [('train', train_labels_path), ('val', val_labels_path), ('test', test_labels_path)]:
        list_file = os.path.join(dataset_path, f'{split}.txt')
        if not os.path.exists(list_file):
            print(f"Warning: {list_file} not found. Skipping {split} labels organization.")
            continue

        with open(list_file, 'r') as f:
            filenames = f.read().splitlines()

        print(f"Organizing {split} labels...")
        for filename in filenames:
            # Assuming image filenames are like '00000001.jpg' and label filenames are '00000001.txt'
            label_filename = os.path.splitext(os.path.basename(filename))[0] + '.txt'
            src_label_path = os.path.join(labels_path, label_filename)
            dest_label_path = os.path.join(dest_path, label_filename)

            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)
            else:
                print(f"Warning: Label file {src_label_path} not found. Skipping.")
    print("Label organization complete.")

if __name__ == '__main__':
    # Assuming the script is run from the project root directory
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    organize_labels(dataset_path)