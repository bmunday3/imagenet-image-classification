import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
import sys
import shutil
from joblib import dump, load


import numpy as np
import os
# from models import *
# from torchvision.models import *
from tqdm import tqdm

from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import distance

import tarfile

def getNextFileName(output_folder):
    highest_num = 0
    for f in os.listdir(output_folder):
        if os.path.isfile(os.path.join(output_folder, f)):
            file_name = os.path.splitext(f)[0]
            try:
                file_num = int(file_name)
                if file_num > highest_num:
                    highest_num = file_num
            except ValueError:
                'The file name "%s" is not an integer. Skipping' % file_name

    output_file_name = str(highest_num + 1)
    return output_file_name


class DriftDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name).convert("RGB")
                
        if self.transform:
            image = self.transform(image)

        return image

class Old_Processing:
    def __init__(self, ds_path, gpu, out_dir, weights, arch, classes, tsfms):
        os.environ["CUDA_VISIBLE_DEVICES"]= gpu

        self.OLD_DATA = ds_path
        self.WEIGHTS = weights
        self.OLD_FEATURES_DIR = "./temp2/"
        self.OLD_DISTANCE_DIR = out_dir
        self.ARCHITECTURE = arch
        self.CLASSES = classes
        self.TSFMS = tsfms

        if not os.path.exists(self.OLD_FEATURES_DIR):
            os.makedirs(self.OLD_FEATURES_DIR)
        if not os.path.exists(os.path.join(self.OLD_FEATURES_DIR,'train_1')):
            os.makedirs(os.path.join(self.OLD_FEATURES_DIR,'train_1'))
        if not os.path.exists(os.path.join(self.OLD_FEATURES_DIR,'train_2')):
            os.makedirs(os.path.join(self.OLD_FEATURES_DIR,'train_2'))
        if not os.path.exists(os.path.join(self.OLD_FEATURES_DIR,'train_3')):
            os.makedirs(os.path.join(self.OLD_FEATURES_DIR,'train_3'))
        if not os.path.exists(os.path.join(self.OLD_FEATURES_DIR,'train_4')):
            os.makedirs(os.path.join(self.OLD_FEATURES_DIR,'train_4'))
        if not os.path.exists(self.OLD_DISTANCE_DIR):
            os.makedirs(self.OLD_DISTANCE_DIR)

    def extract(self):
        print("Extracting features")

        for sd in tqdm(os.listdir(self.OLD_DATA)):
            l1_conv1_feats    = []
            l2_conv1_feats    = []
            l3_conv1_feats    = []
            l4_conv1_feats    = []

            def hook_l1_conv1(module, input, output):
                l1_conv1_feats.extend(output.cpu())
            def hook_l2_conv1(module, input, output):
                l2_conv1_feats.extend(output.cpu())
            def hook_l3_conv1(module, input, output):
                l3_conv1_feats.extend(output.cpu())
            def hook_l4_conv1(module, input, output):
                l4_conv1_feats.extend(output.cpu())

            # load model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            net = models.resnet50()

            # Freeze model parameters
            for param in net.parameters():
                param.requires_grad = False

            # Modify the final layer
            fc_inputs = net.fc.in_features
            net.fc = nn.Sequential(
                nn.Linear(fc_inputs, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )

            net.load_state_dict(torch.load(self.WEIGHTS, map_location=torch.device(device)))
            net = net.to(device)

            with torch.no_grad():
                net.eval()    

            # if self.ARCHITECTURE == "18":
            #     net = models.resnet18(num_classes=int(self.CLASSES))
            # elif self.ARCHITECTURE == "34":
            #     net = models.resnet34(num_classes=int(self.CLASSES))
            # elif self.ARCHITECTURE == "50":
            #     net = models.resnet50(num_classes=int(self.CLASSES))
            # elif self.ARCHITECTURE == "101":
            #     net = models.resnet101(num_classes=int(self.CLASSES))
            # elif self.ARCHITECTURE == "152":
            #     net = models.resnet152(num_classes=int(self.CLASSES))
            # elif self.ARCHITECTURE == "pretrained":
            #     net = models.resnet50(pretrained=True)
            # else:
            #     raise Exception("Invalid model architecture")
            # if self.ARCHITECTURE != "pretrained": 
            #     checkpoint = torch.load(self.WEIGHTS)
            #     # print(checkpoint.keys())
            #     # import sys
            #     # sys.exit()
            #     net.load_state_dict(checkpoint['net'])
            # net.eval()

            old_transforms = self.TSFMS
            old_data = DriftDataset(self.OLD_DATA + '/' +sd, transform=old_transforms)
            loader = torch.utils.data.DataLoader(old_data, batch_size=32, shuffle=False, num_workers=2)

            # registers hooks for feature extraction
            net.layer1[-1].conv1.register_forward_hook(hook_l1_conv1)
            net.layer2[-1].conv1.register_forward_hook(hook_l2_conv1)
            net.layer3[-1].conv1.register_forward_hook(hook_l3_conv1)
            net.layer4[-1].conv1.register_forward_hook(hook_l4_conv1)

            # extract features
            with torch.no_grad():
                for batch_idx, inputs in tqdm(enumerate(loader)):
                    inputs = inputs.to(device)
                    outputs = net(inputs)
            
            # squeeze features, avgpool, flatten, to numpy
            #TODO make this pooling take place inside hook
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            #TODO remove cpu()
            l1_conv1_feats_saf = np.asarray([avgpool(l1_conv1_feats[i]).flatten().numpy() for i in range(len(l1_conv1_feats))])
            l2_conv1_feats_saf = np.asarray([avgpool(l2_conv1_feats[i]).flatten().numpy() for i in range(len(l2_conv1_feats))])
            l3_conv1_feats_saf = np.asarray([avgpool(l3_conv1_feats[i]).flatten().numpy() for i in range(len(l3_conv1_feats))])
            l4_conv1_feats_saf = np.asarray([avgpool(l4_conv1_feats[i]).flatten().numpy() for i in range(len(l4_conv1_feats))])

            np.save(os.path.join(self.OLD_FEATURES_DIR, 'train_1', sd+'.npy'), l1_conv1_feats_saf)
            np.save(os.path.join(self.OLD_FEATURES_DIR, 'train_2', sd+'.npy'), l2_conv1_feats_saf)
            np.save(os.path.join(self.OLD_FEATURES_DIR, 'train_3', sd+'.npy'), l3_conv1_feats_saf)
            np.save(os.path.join(self.OLD_FEATURES_DIR, 'train_4', sd+'.npy'), l4_conv1_feats_saf)

    def distance(self):
        print("Computing distances")
        holder = self.OLD_DISTANCE_DIR
        for layer in [1, 2, 3, 4]:
            print(f"Computing layer {layer}")
            #hack to allow loader to work properly in numpy
            np_load_old = np.load
            # modify the default parameters of np.load
            np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
            # loads train features
            old_features = np.asarray([np.load(os.path.join(self.OLD_FEATURES_DIR, 'train_'+str(layer), f)) 
                                        for f in os.listdir(os.path.join(self.OLD_FEATURES_DIR, 'train_'+str(layer)))])
            # restore np.load for future normal usage
            np.load = np_load_old
            # calculate all classes' means
            #TODO optimize this loop
            class_means = np.asarray([np.mean(old_features[i], axis=0) 
                                        for i in range(len(old_features))])
            # calculate covariance
            X = np.vstack(old_features)
            covariance = EmpiricalCovariance().fit(X)
            maha_dist_array = np.zeros(X.shape[0])

            # loads a single train class's features at a time to prevent memory overflow
            for train_class in os.listdir(os.path.join(self.OLD_FEATURES_DIR, 'train_'+str(layer))):
                Z = np.load(os.path.join(self.OLD_FEATURES_DIR, 'train_'+str(layer), train_class))

                diff = class_means[np.newaxis, :, :] - Z[:, np.newaxis, :]
                mh_class_distances_batch = np.sqrt(
                    np.einsum("jil,jil->ij", np.tensordot(diff, covariance.precision_, axes=(2, 0)), diff)
                )
                mh_extrema_diff_per_batch = np.abs(
                    np.amax(mh_class_distances_batch, axis=0) - np.amin(mh_class_distances_batch, axis=0)
                )
                self.OLD_DISTANCE_DIR = os.path.join('./', 'tempD')
                if not os.path.exists(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer))):
                    os.makedirs(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer)))
                
                np.save(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer), str(train_class)), np.asarray(mh_extrema_diff_per_batch))

            fnames = [np.load(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer), f)) for f in os.listdir(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer)))]
            train_dist = np.hstack(fnames)
            np.save(os.path.join(self.OLD_DISTANCE_DIR, 'train_'+str(layer) + '.npy'), train_dist)
            
            dump(covariance, os.path.join(self.OLD_DISTANCE_DIR, f"c_{layer}.joblib"))
            np.save(os.path.join(self.OLD_DISTANCE_DIR, f"ocm_{layer}.npy"), class_means)  

        
        [shutil.rmtree(os.path.join(self.OLD_DISTANCE_DIR,'train_'+str(l))) for l in [1,2,3,4]] 
        shutil.rmtree(self.OLD_FEATURES_DIR)
        self.OLD_DISTANCE_DIR = holder
    
    def compress(self):
        output_filename = os.path.join(self.OLD_DISTANCE_DIR, "model_info.tar.gz")
        source_dir = os.path.join(self.OLD_DISTANCE_DIR)
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname='.')
        shutil.rmtree(os.path.join('./', 'tempD'))
        
        

if __name__ == "__main__":
    # python image-classification-drift-detection/train_extract_features.py imagenet_10/train/ 4 imagenet_10/old_distances imagenet_10/ckpt.pth 50 10
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path", help="path to old dataset")
    parser.add_argument("gpu", help="gpu to use")
    parser.add_argument("out_dir", help="where distances should be saved")
    parser.add_argument("weights", help="path to weights file, if using pretrained resnet50 you can type anything")
    parser.add_argument("architecture", help="version of resnet, options are 18, 34, 50, 101, 152 or pretrained for a pretrained resnet50")
    parser.add_argument("classes", help="num of classes in the final layer of the model")
    args = parser.parse_args()

    op = Old_Processing(args.ds_path, args.gpu, args.out_dir, args.weights, args.architecture, args.classes)
    op.extract()
    op.distance()