from pathlib import Path
import os
import shutil

os.chdir('ov9d')

train_dir = Path('train')
train_dir.mkdir(parents=True, exist_ok=True)
test_dir = Path('test')
test_dir.mkdir(parents=True, exist_ok=True)
single = Path('oo3d9dsingle')

test_cats = ['bowl', 'box', 'carrot', 'doll', 'gloves', 
             'kite', 'package', 'pear']

[(test_dir/cat).mkdir(parents=True, exist_ok=True) for cat in test_cats]
(test_dir/'all').mkdir(parents=True, exist_ok=True)

for folder in os.listdir(single):
    print(folder)
    cat = '_'.join(folder.split('_')[:-2])
    src_path = (single / folder).resolve()

    if cat in test_cats:
        shutil.move(str(src_path), str((test_dir/cat/folder).resolve()))
        shutil.move(str((test_dir/cat/folder).resolve()), str((test_dir/'all'/folder).resolve()))
    else:
        shutil.move(str(src_path), str((train_dir/folder).resolve()))
