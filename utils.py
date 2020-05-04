##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torchvision.transforms import *
from encoding.transforms.transforms import *

def get_transform(dataset, base_size=None, crop_size=224, etrans=True, **kwargs):
    assert dataset == 'imagenet'
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    base_size = base_size if base_size is not None else int(1.0 * crop_size / 0.875)
    if dataset == 'imagenet':
        train_transforms = []
        val_transforms = []
        if etrans:
            train_transforms.extend([
                ERandomCrop(crop_size),
            ])
            val_transforms.extend([
                ECenterCrop(crop_size),
            ])
            
        else:
            train_transforms.extend([
                RandomResizedCrop(crop_size),
            ])
            val_transforms.extend([
                Resize(base_size),
                CenterCrop(crop_size),
            ])
        train_transforms.extend([
            RandomHorizontalFlip(),
            ToTensor(),
            #Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        val_transforms.extend([
            ToTensor(),
            normalize,
        ])
        transform_train = Compose(train_transforms)
        transform_val = Compose(val_transforms)
    return transform_train, transform_val

def subsample_dataset(total_set, n_splits, split_idx, reduced_size=60000):

    def get_stratified_split_samplers(dataset, n_splits=1, split_idx=0, test_size=None, split_ratio=0.8,
                                      targets=None):
        from sklearn.model_selection import StratifiedShuffleSplit
        if test_size is None:
            test_size = int((1.0 - split_ratio) * len(dataset))
        targets = targets if targets else [x[1] for x in dataset]
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
        sss = sss.split(list(range(len(dataset))), targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        return SimpleSampler(train_idx), SimpleSampler(valid_idx)

    print(f'Using Reduced ImageNet Dataset with size {reduced_size}, now spliting the data ...')
    targets = total_set.targets
    #idx120 = sorted(random.sample(list(range(1000)), k=120))
    idx120 = [15, 22, 42, 65, 66, 70, 91, 101, 103, 114, 120, 127, 128, 129, 130, 132, 134, 140,
              153, 162, 185, 192, 200, 206, 209, 222, 228, 237, 249, 257, 259, 261, 263, 267, 268,
              271, 288, 305, 315, 325, 332, 334, 336, 345, 363, 384, 385, 390, 392, 413, 416, 417,
              419, 434, 446, 453, 461, 466, 475, 484, 490, 492, 516, 518, 552, 558, 560, 563, 566,
              568, 569, 576, 579, 580, 584, 591, 592, 600, 611, 612, 614, 623, 626, 627, 631, 640,
              651, 665, 667, 673, 677, 678, 681, 689, 691, 699, 716, 744, 745, 770, 798, 804, 815,
              821, 824, 825, 836, 847, 863, 891, 928, 939, 945, 946, 952, 955, 970, 982, 993, 998]
    # filter out irrelevant classes
    valid_idx = list(filter(lambda x: targets[x] in idx120, list(range(len(targets)))))
    new_targets = [targets[idx] for idx in valid_idx]

    # pick 120 classes only
    total_set = SampledDataset(total_set, SimpleSampler(valid_idx))
    # pick the reduced_size samples
    reduced_sampler, _ = get_stratified_split_samplers(total_set, 1, 0, len(total_set) - reduced_size,
                                                       targets=new_targets)
    reduced_set = SampledDataset(total_set, reduced_sampler)
    new_targets = [new_targets[idx] for idx in reduced_sampler]
    # 4:1 train and val split
    train_sampler, val_sampler = get_stratified_split_samplers(reduced_set, n_splits, split_idx,
                                                               reduced_size//5, targets=new_targets)
    train_set = SampledDataset(total_set, train_sampler)
    val_set = SampledDataset(total_set, val_sampler)
    return train_set, val_set


class SampledDataset(torch.utils.data.Dataset):
    """Dataset with elements chosen by a sampler"""
    def __init__(self, dataset, sampler):
        self._dataset = dataset
        self._sampler = sampler
        self._indices = list(iter(sampler))

    def __len__(self):
        return len(self._sampler)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

class SimpleSampler(object):
    """Samples elements from [start, start+length) randomly without replacement.
    Parameters
    ----------
    length : int
        Length of the sequence.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
