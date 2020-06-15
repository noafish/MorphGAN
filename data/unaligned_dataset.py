import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import time
import ntpath
import numpy as np
import torch
import torchvision.transforms as transforms


class UnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.fineSize = opt.fineSize
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.transform = get_transform(opt)
        self.nintrm = opt.nintrm


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_B = random.randint(0, self.A_size - 1)
        B_path = self.A_paths[index_B]

        A = self.load_image(A_path)
        B = self.load_image(B_path)
        indxs = np.random.randint(0, self.A_size, self.nintrm)
        sz = A.size()
        reals = torch.Tensor(self.nintrm, sz[0], sz[1], sz[2])
        for idx, i in enumerate(indxs):
            real = self.load_image(self.A_paths[i])
            reals[idx] = real

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path,
                 'reals': reals}


    def load_image(self, im_path):
        A_img = Image.open(im_path).convert('RGB')
        sz = A_img.size
        szm = min(sz)
        self.orig_size = sz

        A = self.transform(A_img)
        if self.opt.nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)
        return A



    def __len__(self):
        return self.A_size

    def name(self):
        return 'UnalignedDataset'

