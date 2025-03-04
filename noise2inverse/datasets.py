#from noise2inverse import tiffs
#from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from torch.utils.data import (
#    DataLoader,
    Dataset
)
import tifffile
import os
from torchvision.transforms.functional import crop as tv_crop


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
    def __init__(self, *datasets, strategy="X:1", test = False, crop_size=None, center_weight=0.0, num_crops=1):
        super(Noise2InverseDataset, self).__init__()

        self.datasets = datasets #list of datasets for each split
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

        # Crop parameters
        self.crop_size = crop_size
        self.center_weight = center_weight
        self.num_crops = num_crops

    @property
    def num_splits(self):
        return len(self.datasets)

    @property
    def num_slices(self): #total number of slices so is img_num * split
        return len(self.datasets[0])

    def __getitem__(self, i):
        # set crop index
        non_crop_idex = i // self.num_crops #get the index of the original dataset length (no of slices * no of splits)
        
        num_splits = self.num_splits
        slice_idx = non_crop_idex // num_splits #get the slice index so that every n index is the same slice but different split so different input and target
        split_idx = non_crop_idex % num_splits #get the split index, ie which subset, 

        input_idxs = self.input_idxs[split_idx]
        target_idxs = self.target_idxs[split_idx]

        slices = [ds[slice_idx] for ds in self.datasets] #take the slices in each split (dataset), read the file from __getitem__ TiffDataset




        # crop the slices 
        if self.crop_size is not None:
            # would like to add fixed size cropping
            if self.center_weight == -1: #TODO: this is used for testing purposea
                # apply a fixed point crop 
                c, h, w = slices[0].shape
                crop_h = self.crop_size
                crop_w = self.crop_size

                offset_index = i % self.num_crops

                # list of 5 offsets

                offset_y = [h//2-self.crop_size//2, h//2-self.crop_size//2-self.crop_size, h//2-self.crop_size//2-self.crop_size, h//2+self.crop_size//2, h//2+self.crop_size//2]
                offset_x = [w//2-self.crop_size//2, w//2-self.crop_size//2-self.crop_size, w//2+self.crop_size//2, w//2-self.crop_size//2-self.crop_size, w//2+self.crop_size//2]

                cropped_slices = [self._apply_crop(s, offset_y[offset_index], offset_x[offset_index], crop_h, crop_w)
                                    for s in slices]





            # Convert each slice to a numpy array if needed
            # directory operate on the tensor
            # If we want the same random crop across all slices,
            # we need to compute a single offset from the first slice's shape.
            elif (self.crop_size is not None) and (len(slices) > 0):
                c, h, w = slices[0].shape
                crop_h = self.crop_size
                crop_w = self.crop_size
                if (crop_h > 0 and crop_w > 0 and crop_h <= h and crop_w <= w):
                    offset_y, offset_x = self._compute_random_offset(h, w, crop_h, crop_w)
                    # Crop all slices with the same offset
                    cropped_slices = [self._apply_crop(s, offset_y, offset_x, crop_h, crop_w)
                                    for s in slices]

            inputs = [cropped_slices[j] for j in input_idxs] #take the slice from the input splits
            targets = [cropped_slices[j] for j in target_idxs] #take the slice from the target splits

        else:
            inputs = [slices[j] for j in input_idxs] #take the slice from the input splits
            targets = [slices[j] for j in target_idxs]

        inp = torch.mean(torch.stack(inputs), dim=0)
        tgt = torch.mean(torch.stack(targets), dim=0)

        return inp, tgt

    def __len__(self):
        return self.num_splits * self.num_slices * self.num_crops

    def _compute_random_offset(self, h, w, crop_h, crop_w):
        # Calculate the valid range for top-left corner of the crop
        max_y = h - crop_h
        max_x = w - crop_w

        # The center offset
        center_y = (h - crop_h) // 2
        center_x = (w - crop_w) // 2

        # The random offset (uniform)
        rand_y = np.random.randint(0, max_y + 1)
        rand_x = np.random.randint(0, max_x + 1)

        # Interpolate between center and random offset
        offset_y = int((1 - self.center_weight) * rand_y + self.center_weight * center_y)
        offset_x = int((1 - self.center_weight) * rand_x + self.center_weight * center_x)
        return offset_y, offset_x

    def _apply_crop(self, img, offset_y, offset_x, crop_h, crop_w):
        return tv_crop(img, offset_y, offset_x, crop_h, crop_w)

