# Learning What and Where to Transfer (ICML 2019)
Learning What and Where to Transfer (ICML 2019)
https://arxiv.org/abs/1905.05901

## Requirements

- `python>=3.6`
- `pytorch>=1.0`
- `torchvision`
- `cuda>=9.0`

**Note.** The reported results in our paper were obtained in the old-version pytorch (`pytorch=1.0`, `cuda=9.0`). We recently executed again the experiment commands as described below using the recent version (`pytorch=1.6.0`, `torchvision=0.7.0`, `cuda=10.1`), and obtained similar results as reported in the paper.

## Prepare Datasets

You can download CUB-200 and Stanford Dogs datasets
- CUB-200: from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Stanford Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/

You need to run the below pre-processing script for DataLoader.

```bash
python cub200.py /data/CUB_200_2011
python dog.py /data/dog
```

## Train L2T-ww

You can train L2T-ww models with the same settings in our paper.

```bash
python train_l2t_ww.py --dataset cub200 --datasplit cub200 --dataroot /data/CUB_200_2011
python train_l2t_ww.py --dataset dog --datasplit dog --dataroot /data/dog
python train_l2t_ww.py --dataset cifar100 --datasplit cifar100 --dataroot /data/ --experiment logs/cifar100_0/ --source-path logs --source-model resnet32 --source-domain tinyimagenet-200 --target-model vgg9_bn --pairs 4-0,4-1,4-2,4-3,4-4,9-0,9-1,9-2,9-3,9-4,14-0,14-1,14-2,14-3,14-4 --batchSize 128
python train_l2t_ww.py --dataset stl10 --datasplit stl10 --dataroot /data/ --experiment logs/stl10_0/ --source-path logs --source-model resnet32 --source-domain tinyimagenet-200 --target-model vgg9_bn --pairs 4-0,4-1,4-2,4-3,4-4,9-0,9-1,9-2,9-3,9-4,14-0,14-1,14-2,14-3,14-4 --batchSize 128
```

