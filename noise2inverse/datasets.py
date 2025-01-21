from noise2inverse import tiffs
from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from torch.utils.data import (
    DataLoader,
    Dataset
)
import tifffile
import pdb
import os


class TiffDataset(Dataset):
    """Documentation for TiffDataset

    """
    def __init__(self, img_path, img_list, channel=1, test=False):
        super(TiffDataset, self).__init__()
        #concatenate the path and the image name
        with open(img_list, 'r') as f:
            img_names = f.read().splitlines()
        self.paths = [os.path.join(img_path,img) for img in img_names]
        self.channel = channel
        self.test = test

    def __getitem__(self, i):
        try:
            if self.channel == 1:
                img = tifffile.imread(str(self.paths[i])).astype(np.float32)
                if img.ndim == 2:
                    img = img[None, ...]
            else: #stacking 2.5D images
                img_list = []
                for j in range(i - self.channel//2, i + self.channel//2 + 1):
                    if j < 0:
                        img_temp = tifffile.imread(str(self.paths[0])).astype(np.float32)
                        if img_temp.ndim == 2:
                            img_temp = img_temp[None, ...]
                        img_list.append(img_temp)
                    elif j >= len(self.paths):
                        img_temp = tifffile.imread(str(self.paths[i])).astype(np.float32)
                        if img_temp.ndim == 2:
                            img_temp = img_temp[None, ...]
                        img_list.append(img_temp)
                    else:
                        img_temp = tifffile.imread(str(self.paths[j])).astype(np.float32)
                        if img_temp.ndim == 2:
                            img_temp = img_temp[None, ...]
                        img_list.append(img_temp)
                img = np.vstack(img_list)
        except Exception as e:
            print(e)
            print(self.paths[i])
            #pdb.set_trace()


        return torch.from_numpy(img)

    def __len__(self):
        return len(self.paths)


class SupervisedDataset(Dataset):
    """Documentation for SupervisedDataset

    """
    def __init__(self, input_ds, target_ds):
        super(SupervisedDataset, self).__init__()
        self.input_ds = input_ds
        self.target_ds = target_ds

        assert len(input_ds) == len(target_ds)

    def __getitem__(self, i):
        return self.input_ds[i], self.target_ds[i]

    def __len__(self):
        return len(self.input_ds)


class Noise2InverseDataset(Dataset):
    """Documentation for Noise2InverseDataset

    """
    def __init__(self, *datasets, strategy="X:1", test = False):
        super(Noise2InverseDataset, self).__init__()

        self.datasets = datasets
        max_len = max(len(ds) for ds in datasets)
        min_len = min(len(ds) for ds in datasets)

        assert min_len == max_len #checking each split has equal number of slices

        assert strategy in ["X:1", "1:X"]
        self.strategy = strategy

        if strategy == "X:1":
            num_input = self.num_splits - 1
        else:
            num_input = 1

        self.test = test

        # For num_splits=4, 1:X, we have
        # input_idxs =  [(0,),      (1,),      (2,),      (3,)]
        # target_idxs = [{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}]
        split_idxs = set(range(self.num_splits))
        self.input_idxs = list(combinations(split_idxs, num_input))
        self.target_idxs = [split_idxs - set(idxs) for idxs in self.input_idxs]

    @property
    def num_splits(self):
        return len(self.datasets)

    @property
    def num_slices(self): #total number of slices so is img_num * split
        return len(self.datasets[0])

    def __getitem__(self, i):
        
        num_splits = self.num_splits
        slice_idx = i // num_splits #get the slice index
        split_idx = i % num_splits #get the split index, ie which subset

        input_idxs = self.input_idxs[split_idx]
        target_idxs = self.target_idxs[split_idx]

        slices = [ds[slice_idx] for ds in self.datasets] #take the slice in all split, read the file
        inputs = [slices[j] for j in input_idxs] #take the slice from the input splits
        targets = [slices[j] for j in target_idxs]

        inp = torch.mean(torch.stack(inputs), dim=0)
        tgt = torch.mean(torch.stack(targets), dim=0)

        return inp, tgt

    def __len__(self):
        return self.num_splits * self.num_slices
