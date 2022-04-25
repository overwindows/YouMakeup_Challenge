# YouMakeUp MTVG

## Introduction

This is baseline code for the sub-task of [YouMakeup_Challenge_2022](https://github.com/AIM3-RUC/YouMakeup_Challenge_2022) to localize the target event in the video. We applied [MMN](https://arxiv.org/pdf/2109.04872v2.pdf) to [YouMakeup](https://github.com/AIM3-RUC/YouMakeup) dataset. You can get the details of the model from the paper: [Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding](https://arxiv.org/pdf/2109.04872v2.pdf)

## Preparation

Environment: <br>

Linux,  GCC>=5.4, CUDA >= 9.2,

Python>=3.6, PyTorch>=1.1.0,



1.Download video feature

c3d_rgb: [google drive](https://drive.google.com/open?id=1gPGEYej70hKM6e-ftXI0RBNzn4AokMJ1) or [baidu_cloud](https://pan.baidu.com/s/1zaKC2BIw5ARmuYKybcAgDg)  password:hcw8

i3d_rgb: [google drive](https://drive.google.com/open?id=1cT5MKcmSmqS6xC_i2dI2wbJ3n7mdFh7o) or [baidu_cloud](https://pan.baidu.com/s/1OH_6LvUWvRTcPO33wcjZ_g)  password:nrlu

The hdf5 files should be put into folder `./dataset/makeup`.



2.Prepare caption data

**Annotations of videos** on train/val set are [here](https://drive.google.com/drive/folders/1vKTZh8eDAJYAtD6AYa7606xu74azd2iY?usp=sharing) or [BaiduNetDisk](), password:.

Caption data has been provided in folder `./dataset/makeup`.

```bash
#Caption data has been provided. You don't have to run this code.

python ./dataset/makeup/prepare_data.py
```



## Training and Validation

Training

```bash
./scripts/makeup_train.sh
```

Evaluation

```
./scripts/eval.sh
```

You can verify the config files or bash files to select the appropriate settings.

The default feature we use is I3D feature. If you want to change to C3D feature, you need to change the dataset directory in `./mmn/pahts_catalog.py`



## Checkpoints are available

i3d_e20.model: [google_drive]() or [baidu_cloud]()

c3d_e20.model: [google_drive]() or [baidu_cloud]()

The checkpoint should be put into `./outputs/pool_makeup_i3d/` or `./outputs/pool_makeup_c3d/`.

The checkpoints' performances are as below:

| feature | R@1,IoU@0.3 | R@1,IoU@0.5 | R@1,IoU@0.7 | R@5,IoU@0.3 | R@5,IoU@0.5 | R@5,IoU@0.7 |
| :-----: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| **C3D** |    33.47    |    23.05    |    11.78    |    63.28    |    48.88    |    25.07    |
| **I3D** |    48.09    |    35.18    |    20.08    |    76.79    |    64.13    |    36.25    |



## Citation

```
@article{DBLP:journals/corr/abs-2109-04872,
  author    = {Zhenzhi Wang and
               Limin Wang and
               Tao Wu and
               Tianhao Li and
               Gangshan Wu},
  title     = {Negative Sample Matters: {A} Renaissance of Metric Learning for Temporal
               Grounding},
  journal   = {CoRR},
  volume    = {abs/2109.04872},
  year      = {2021}
}
```
