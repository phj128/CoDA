# CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects
### [Project Page](https://phj128.github.io/page/CoDA/index.html) | [Paper](https://arxiv.org/abs/2505.21437)

> CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects  
> [Huaijin Pi](https://phj128.github.io/),
[Zhi Cen](https://anitacen.github.io/),
[Zhiyang Dou](https://frank-zy-dou.github.io/),
[Taku Komura](https://i.cs.hku.hk/~taku) \
> NeurIPS 2025

<p align="center">
    <img src=docs/image/teaser_v2.png />
</p>

## News
[October 13, 2025] Training and Evaluation code released. 

## TODOs

- [x] Release training code. 

- [x] Release evaluation code. 

## Dependencies
To create the environment, follow the instructions:

1. Create new conda environment and install pytroch:

```
conda create -n coda python=3.10
pip install -r requirements.txt
pip install -e .
```

2. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). 

3. Download CLIP (```clip-vit-base-patch32```) and glove checkpoints (refer to this [link](https://github.com/GuyTevet/motion-diffusion-model/blob/main/prepare/download_glove.sh)).

4. Download [ARCTIC](https://arctic.is.tue.mpg.de/) dataset and [GRAB](https://grab.is.tue.mpg.de/) dataset.

5. Download preprocessed data from [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3011165_connect_hku_hk/EdKhNWsKg2hClBsD0FAiPxIB9y8v6HfMZQSLGpr0D7AcYQ?e=uwiTnD).

Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites. 
*We're unable to provide the original data due to the license restrictions.* 
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

6. Download our pretrained checkpoints from [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3011165_connect_hku_hk/EfQOTeVxkJFGusThZkxjNbMB4Y2PVBTwHVFEGsc9dLiC-Q?e=slnIB5) and evaluator from [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3011165_connect_hku_hk/EXw_bVquOWxDlLLmGkmw06YBsmariT-CFuvJf7MyEtZHOA?e=7wxuiU).

7. Rename these downloaded files and organize them following the file structure:

```
inputs
├── checkpoints
│   ├── body_models/smplx/
│   │   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
│   ├── body_models/smpl/
│   │   └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)
│   ├── glove
│   ├── huggingface
│   │   └── clip-vit-base-patch32
│   └── arcticobj
├── amass
├── arctic
├── arctic_neutral
├── grab_extracted
├── grab_neutral
└── release_checkpoints
```

8. Calculate the corresponding bps representation.
```
python tools/preprocess/arctic_bps.py
python tools/preprocess/grab_bps.py
```

## Evaluation
Test with our provided checkpoints.
```
python tools/train.py exp=wholebody/obj_arctic global/task=wholebody/test_arctic
```

## Training
1. Train the object trajectory model.
```
python tools/train.py exp=objtraj/arctic
```

2. Train the end-effector trajectory model.
```
python tools/train.py exp=ee/arctic
```

3. Train separate motion diffusion models.
```
python tools/train.py exp=handpose/lefthand_mixed
python tools/train.py exp=handpose/righthand_mixed
python tools/train.py exp=bodypose/mixed
```

After training, please assign the corresponding checkpoints path in `coda/configs/global/task/wholebody/test_arctic.yaml` for further evaluation.

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{pi2025coda,
  title={CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects},
  author={Pi, Huaijin and Cen, Zhi and Dou, Zhiyang and Komura, Taku},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

# Acknowledgement

We thank the authors of
[GVHMR](https://github.com/zju3dv/GVHMR),
[HGHOI](https://github.com/zju3dv/hghoi),
and [DNO](https://github.com/korrawe/Diffusion-Noise-Optimization)
for their great works, without which our project/code would not be possible.

