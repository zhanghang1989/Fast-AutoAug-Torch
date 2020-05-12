# Fast-AutoAug-Torch

Search for [Fast AutoAugment](https://arxiv.org/abs/1905.00397) using [AutoTorch](http://autotorch.org/). [AutoAugment](https://arxiv.org/abs/1805.09501) and [RandAugment](https://arxiv.org/abs/1909.13719) are also implemented for comparison. 

This example will be used in the tutorial on [From HPO to NAS: Automated Deep Learning](https://hangzhang.org/CVPR2020/) at CVPR 2020.

| model | augment| epoch | Acc | weights |
|-------|--------|-------|-----|---------|
|ResNet-50| baseline | 120 | 76.48 |
|ResNet-50| AA | 120 | 76.66 |
|ResNet-50| Fast AA | 120| 76.88 |
|ResNet-50| Rand AA | 120 | 76.79 |
|ResNet-50| baseline | 270| 77.17 |
|ResNet-50| AA | 270 | 77.78 | [link](https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_aa-0cb27f8e.pth) |
|ResNet-50| Fast AA | 270| 77.73 | [link](https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_fast_aa-3342410e.pth) |
|ResNet-50| Rand AA | 270| **77.97** | [link](https://hangzh.s3-us-west-1.amazonaws.com/others/resnet50_rand_aug-e38097c7.pth) |

Approaches:
``
AA: AutoAugment,
Fast AA: Fast AutoAugment,
Rand AA: RandAugment,
``

Training HP setting:
``
learning rate: 0.2,
batch size: 512,
weight decay: 1e-4,
``

Fast AA setting (using random search):
``
K=8, T=1, N=5,
``
resulting 40 subpolicies in total.

RandAug setting (after grid search):
``
n=2, m=12,
``


## Setup
### Install dependencies
- PyTorch, please follow the [instructions](https://pytorch.org/get-started/locally/).

- Install AutoTorch and PyTorch-Encoding toolkits:

```bash
pip install autotorch --pre
pip install torch-encoding --pre
```

- Install Apex (optional):

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
### Prepare dataset

```bash
# assuming you have downloaded the ImageNet dataset in the current folder
python prepare_imagenet.py --download-dir ./
```

## Fast AutoAugment Training
### Search policy
If you want to skip this step, you can download the searched policy [here](https://hangzh.s3-us-west-1.amazonaws.com/others/imagenet_policy.at).

```bash
python search_policy.py --reduced-size 60000 --epochs 120 --nfolds 8 --num-trials 200  --save-policy imagenet_policy.at
```

[An example learned policy](./imagenet_policy.md)

### Train with searched policy

```bash
python train.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 270 --checkname resnet50_fast_aa --lr 0.025 --batch-size 64 --auto-policy imagenet_policy.at
```

## Auto Augment and RandomAugment

### Training with AutoAugment policy
This repo only enables training with searched policy from the original paper. 

```bash
# download the policy
wget https://hangzh.s3-us-west-1.amazonaws.com/others/aa_policy.at
python train.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 270 --checkname resnet50_fast_aa --lr 0.025 --batch-size 64 --auto-policy  aa_policy.at
```

### Training with RandomAugment
```bash
python train.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 270 --checkname resnet50_rand_aug --lr 0.025 --batch-size 64 --rand-aug
```

