##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import logging


import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import multiprocessing as mp
import multiprocessing.pool
try:
    torch.multiprocessing.set_start_method('spawn',force=True)
except RuntimeError:
    pass

import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import autotorch as at
import encoding
from encoding.utils import (accuracy, AverageMeter, LR_Scheduler)
from utils import get_transform, subsample_dataset
from augment import Augmentation, augment_dict

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='FastAA-AutoTorch')
    # model params 
    parser.add_argument('--model', type=str, default='resnet50',
                        help='network model type (default: densenet)')
    # data
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='training dataset (default: imagenet)')
    parser.add_argument('--reduced-size', type=int, default=60000,
                        help='reduced imagenet size for training (default: 60000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--workers', type=int, default=12,
                        help='dataloader threads')
    parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/.encoding/data'),
                        help='data location for training')
    # input size
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    parser.add_argument('--base-size', type=int, default=None,
                        help='base image size')
    # training hp
    parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=5e-5, 
                        help='SGD weight decay (default: 1e-4)')
    # AutoTorch
    parser.add_argument('--nfolds', type=int, default=None,
                        help='Num of split folds')
    parser.add_argument('--num-trials', default=200, type=int,
                        help='number of trail tasks')
    parser.add_argument('--save-policy', type=str, required=True,
                        help='path to auto augment policy')
    parser = parser

    args = parser.parse_args()
    return args

def train_network(args, gpu_manager, split_idx, return_dict):
    gpu = gpu_manager.request()
    print('gpu: {}, split_idx: {}'.format(gpu, split_idx))

    # single gpu training only for evaluating the configurations
    model = encoding.models.get_model(args.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

    model.cuda(gpu)
    criterion.cuda(gpu)

    # init dataloader
    base_size = args.base_size if args.base_size is not None else int(1.0 * args.crop_size / 0.875)
    transform_train, _ = get_transform(
            args.dataset, args.base_size, args.crop_size)
    total_set = encoding.datasets.get_dataset('imagenet', root=args.data_dir,
                                              transform=transform_train, train=True, download=True)
    trainset, valset = subsample_dataset(total_set, args.nfolds, split_idx, args.reduced_size)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True, pin_memory=True)
    # lr scheduler
    lr_scheduler = LR_Scheduler('cos',
                                base_lr=args.lr,
                                num_epochs=args.epochs,
                                iters_per_epoch=len(train_loader),
                                quiet=True)

    # write results into config file
    def train(epoch):
        model.train()
        top1 = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):
            lr_scheduler(optimizer, batch_idx, epoch, 0)
            data, target = data.cuda(gpu), target.cuda(gpu)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    def validate(auto_policy):
        model.eval()
        top1 = AverageMeter()
        _, transform_val = get_transform(args.dataset, args.base_size, args.crop_size)
        if auto_policy is not None:
            transform_val.transforms.insert(0, Augmentation(auto_policy))
        valset.transform = transform_val
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.cuda(gpu), target.cuda(gpu)
            with torch.no_grad():
                output = model(data)
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

        return top1.avg

    for epoch in tqdm(range(0, args.epochs)):
        train(epoch)

    #acc = validate(None)
    #print('baseline accuracy: {}'.format(acc))

    ops = list(augment_dict.keys())
    sub_policy = at.List(
        at.List(at.Choice(*ops), at.Real(0, 1), at.Real(0, 1)),
        at.List(at.Choice(*ops), at.Real(0, 1), at.Real(0, 1)),
    )
    searcher = at.searcher.RandomSearcher(sub_policy.cs)
    # avoid same defaults
    config = searcher.get_config()
    for i in range(args.num_trials):
        config = searcher.get_config()
        auto_policy = sub_policy.sample(**config)
        acc = validate(auto_policy)
        searcher.update(config, acc.item(), done=True)

    gpu_manager.release(gpu)
    topK_cfgs = searcher.get_topK_configs(5)
    policy = [sub_policy.sample(**cfg) for cfg in topK_cfgs]
    return_dict[split_idx] = policy
    #print(f'{split_idx}, searcher._results: {searcher._results}')

def train_network_map(args):
    train_network(*args)

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess

class GPUManager(object):
    def __init__(self, ngpus):
        self._gpus = mp.Manager().Queue()
        for i in range(ngpus):
            self._gpus.put(i)

    def request(self):
        return self._gpus.get()

    def release(self, gpu):
        self._gpus.put(gpu)

def main():
    # temporary solution for imp DeprecationWarning
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    logging.basicConfig(level=logging.DEBUG)

    args = get_args()
    ngpus = torch.cuda.device_count()
    args.nfolds = args.nfolds if args.nfolds else ngpus

    gpu_manager = GPUManager(ngpus)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    tasks = ([args, gpu_manager, i, return_dict] for i in range(args.nfolds))
        
    p = MyPool(processes=ngpus)
    p.map(train_network_map, tasks)

    all_policies = list(return_dict.values())
    policies = []
    for i, policy in enumerate(all_policies):
        policies.extend(policy)
    print('len(policies):', len(policies))
    print('policies: ', policies)
    at.save(policies, args.save_policy)

if __name__ == "__main__":
    main()

