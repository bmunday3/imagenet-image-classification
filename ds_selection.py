import argparse
import os
from shutil import copy
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", help="path to imagenet formatted dataset")
    parser.add_argument("out_dir", default='./', nargs='?', help="where new dataset should be saved")
    args = parser.parse_args()

    ds_path = args.ds_path
    out_dir = args.out_dir
    folders = os.listdir(ds_path)
    selection_amount = int(75 + (1000 - len(folders))/2)

    path = os.path.join(out_dir, "dataset")
    if not os.path.exists(path):
        os.mkdir(path)
    
    for f in tqdm(folders):
        print("Copying ", f)
        old_folder = os.path.join(ds_path, f)
        new_folder = os.path.join(path, f)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        files = os.listdir(old_folder)[:selection_amount]
        [copy(os.path.join(old_folder, fileName), new_folder) for fileName in files]

