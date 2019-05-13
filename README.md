# Learning What and Where to Transfer (ICML 2019)
Learning What and Where to Transfer (ICML 2019)

## Requirements
- pytorch 1.0
- CUDA 9.0
- CUDNN 7

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
```

