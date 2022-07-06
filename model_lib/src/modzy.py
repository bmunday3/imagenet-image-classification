import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms, models
import sys

import numpy as np
import os
import shutil

import random
import struct
from Crypto.Cipher import AES
from Crypto import Random
from datetime import date


class Utilities:
    def install(self, net):
        l1_conv1_feats = []
        l2_conv1_feats = []
        l3_conv1_feats = []
        l4_conv1_feats = []

        def hook_l1_conv1(module, input, output):
            l1_conv1_feats.extend(output.cpu())
        def hook_l2_conv1(module, input, output):
            l2_conv1_feats.extend(output.cpu())
        def hook_l3_conv1(module, input, output):
            l3_conv1_feats.extend(output.cpu())
        def hook_l4_conv1(module, input, output):
            l4_conv1_feats.extend(output.cpu())
            
        # registers hooks for feature extraction
        hook_1 = net.layer1[-1].conv1.register_forward_hook(hook_l1_conv1)
        hook_2 = net.layer2[-1].conv1.register_forward_hook(hook_l2_conv1)
        hook_3 = net.layer3[-1].conv1.register_forward_hook(hook_l3_conv1)
        hook_4 = net.layer4[-1].conv1.register_forward_hook(hook_l4_conv1)

        return net, (l1_conv1_feats, l2_conv1_feats, l3_conv1_feats, l4_conv1_feats), (hook_1,hook_2,hook_3,hook_4)         
    
    def encrypt_file(self, key, in_filename, out_filename=None, chunksize=64*1024):
        """ Encrypts a file using AES (CBC mode) with the
            given key.
            key:
                The encryption key - a bytes object that must be
                either 16, 24 or 32 bytes long. Longer keys
                are more secure.
            in_filename:
                Name of the input file
            out_filename:
                If None, '<in_filename>.enc' will be used.
            chunksize:
                Sets the size of the chunk which the function
                uses to read and encrypt the file. Larger chunk
                sizes can be faster for some files and machines.
                chunksize must be divisible by 16.
        """
        if not out_filename:
            out_filename = in_filename + '.enc'

        iv = os.urandom(16)
        encryptor = AES.new(key, AES.MODE_CBC, iv)
        filesize = os.path.getsize(in_filename)

        with open(in_filename, 'rb') as infile:
            with open(out_filename, 'wb') as outfile:
                outfile.write(struct.pack('<Q', filesize))
                outfile.write(iv)

                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    elif len(chunk) % 16 != 0:
                        chunk += b' ' * (16 - len(chunk) % 16)

                    outfile.write(encryptor.encrypt(chunk))
        return out_filename

    def decrypt_file(self, key, in_filename, out_filename=None, chunksize=24*1024):
        """ Decrypts a file using AES (CBC mode) with the
            given key. Parameters are similar to encrypt_file,
            with one difference: out_filename, if not supplied
            will be in_filename without its last extension
            (i.e. if in_filename is 'aaa.zip.enc' then
            out_filename will be 'aaa.zip')
        """
        if not out_filename:
            out_filename = os.path.splitext(in_filename)[0]

        with open(in_filename, 'rb') as infile:
            origsize = struct.unpack('<Q', infile.read(struct.calcsize('Q')))[0]
            iv = infile.read(16)
            decryptor = AES.new(key, AES.MODE_CBC, iv)

            with open(out_filename, 'wb') as outfile:
                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    outfile.write(decryptor.decrypt(chunk))

                outfile.truncate(origsize)
        return out_filename
    
    def encrypt(self, weights_path, output_name=None):
        key = b'?(v\xed\x16\xb4E(\\\x85\xf5\xa0\xce\xb0\x04\x9e'
        file_name = self.encrypt_file(key, weights_path, output_name)
        return file_name
   
    def decrypt(self, encrypted_path, output_name="weights.pth"):
        t = date.today()
        b = date(2021, 1, 1)
        e = date(2022, 12, 31)
        if (t >= b and t <= e):
            key = b'?(v\xed\x16\xb4E(\\\x85\xf5\xa0\xce\xb0\x04\x9e'
            file_name = self.decrypt_file(key, encrypted_path, output_name)
        else:
            raise Exception("License is expired")
        return file_name

    def load_weights(self, path, classes, architecture):
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

        net.load_state_dict(torch.load(path, map_location=torch.device(device)))
        net = net.to(device)

        with torch.no_grad():
            net.eval()         

        return net, device
    
    def process(self, arrays, new_features_dir):
        features_1, features_2, features_3, features_4 = arrays
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        l1_conv1_feats_saf = np.asarray([avgpool(features_1[i]).flatten().detach().numpy() for i in range(len(features_1))]).tolist()
        l2_conv1_feats_saf = np.asarray([avgpool(features_2[i]).flatten().detach().numpy() for i in range(len(features_2))]).tolist()
        l3_conv1_feats_saf = np.asarray([avgpool(features_3[i]).flatten().detach().numpy() for i in range(len(features_3))]).tolist()
        l4_conv1_feats_saf = np.asarray([avgpool(features_4[i]).flatten().detach().numpy() for i in range(len(features_4))]).tolist()

        return (l1_conv1_feats_saf, l2_conv1_feats_saf, l3_conv1_feats_saf, l4_conv1_feats_saf)

    
