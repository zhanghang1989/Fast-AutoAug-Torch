# Fast-AutoAug-Torch

## Quick Start
### Prepare dataset

```bash
# assuming you have downloaded the dataset in the current folder
python prepare_imagenet.py --download-dir ./
```

### Search policy
```bash
python search_policy.py --reduced-size 60000 --epochs 120 --nfolds 8 --num-trials 200  --save-policy imagenet_policy.at
```

### Train with searched policy
```bash
python train.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_check --lr 0.025 --batch-size 64 --auto-policy imagenet_policy.at
```



