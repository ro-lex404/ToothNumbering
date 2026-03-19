# This is a script to organize the taskdataset into the structure 
# required for training a YOLO model. 
# It will create a new folder called 'dataset' with subfolders for 
# training, validation, and testing data, each containing 'images' and 'labels' 
# folders.

import os
import random
import shutil

# 1. Define your current paths
image_dir = 'ToothNumber_TaskDataset/images/'
label_dir = 'ToothNumber_TaskDataset/labels/'
output_dir = 'dataset/'

# 2. Get all filenames (assuming .jpg and .txt have the same name)
filenames = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(filenames)

# 3. Define split sizes (70% train, 20% val, 10% test)
train_split = int(0.7 * len(filenames))
val_split = int(0.9 * len(filenames))

splits = {
    'train': filenames[:train_split],
    'val': filenames[train_split:val_split],
    'test': filenames[val_split:]
}

# 4. Move the files
for split_name, split_files in splits.items():
    # Create the folders
    os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split_name, 'labels'), exist_ok=True)
    
    for f in split_files:
        # Move image
        shutil.copy(os.path.join(image_dir, f + '.jpg'), 
                    os.path.join(output_dir, split_name, 'images', f + '.jpg'))
        # Move label
        shutil.copy(os.path.join(label_dir, f + '.txt'), 
                    os.path.join(output_dir, split_name, 'labels', f + '.txt'))

print("Data reorganization complete!")