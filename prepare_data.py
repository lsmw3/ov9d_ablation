from pathlib import Path
import os
import shutil
import random

os.chdir('ov9d')

main_folder = "oo3d9dmulti"

train_dir = Path('train_multi')
train_dir.mkdir(parents=True, exist_ok=True)
test_dir = Path('test_multi')
test_dir.mkdir(parents=True, exist_ok=True)
# single = Path('oo3d9dsingle')

# test_cats = ['bowl', 'box', 'carrot', 'doll', 'gloves', 
#              'kite', 'package', 'pear']

# [(test_dir/cat).mkdir(parents=True, exist_ok=True) for cat in test_cats]
# (test_dir/'all').mkdir(parents=True, exist_ok=True)

# for folder in os.listdir(single):
#     print(folder)
#     cat = '_'.join(folder.split('_')[:-2])
#     src_path = (single / folder).resolve()

#     if cat in test_cats:
#         shutil.move(str(src_path), str((test_dir/cat/folder).resolve()))
#         shutil.move(str((test_dir/cat/folder).resolve()), str((test_dir/'all'/folder).resolve()))
#     else:
#         shutil.move(str(src_path), str((train_dir/folder).resolve()))


subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

# Shuffle the subfolders
random.shuffle(subfolders)

# Split into train (80%) and test (20%)
split_idx = int(len(subfolders) * 0.8)
train_subfolders = subfolders[:split_idx]
test_subfolders = subfolders[split_idx:]

# Move subfolders to train and test directories
for folder in train_subfolders:
    shutil.move(os.path.join(main_folder, folder), os.path.join(train_dir, folder))

for folder in test_subfolders:
    shutil.move(os.path.join(main_folder, folder), os.path.join(test_dir, folder))

print(f"Moved {len(train_subfolders)} folders to train and {len(test_subfolders)} folders to test.")