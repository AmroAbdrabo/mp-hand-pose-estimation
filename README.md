# 3D Hand Pose Estimation (Project 1) - Group KIdivergent

## Branch abstract:
- **Model**: Single ResNet101 based approach
- **Optimizer**: Adam with learning rate 1e-3
- **Loss**: 3D joint loss
- **Transforms**: Scale normalization, change background, add clutter, resizing to (128, 128) 
- **Batch size**: Batch size 128, 210 epochs


### Installing requirements

There are additional requirement to be installed via pip:

```python
pip install -r requirements.txt
```

### Downloading 128x128 background images

To get the backgrounds used for data augmentation, download this [dataset](http://chaladze.com/l5/img/Linnaeus%205%20128X128.rar) of 128x128 images and put all the (1200) images from the folder `Linnaeus 5 128X128/train/other/` to a directory, i.e. called **backgrounds**.
```
└───backgrounds
	1_128.jpg
	2_128.jpg
	3_128.jpg
	...
	1200_128.jpg
```

### Environment variables

Make sure to have these environments variables:

- `MP_BACKGROUNDS` : This should point to the directory **backgrounds** which stores all the background images, with no trailing slash:
```python
MP_BACKGROUNDS="$HOME/project1/backgrounds"
```

- `MP_EXPERIMENTS`: This should point to the directory where you will output experiments runs (folders which contain the saved state of the model and optimizer)

- `MP_DATA`: This should point to the directory `freihand_dataset_MP` folder (located in /cluster/project/infk/hilliges/lectures/mp21/project1/ on the cluster).
