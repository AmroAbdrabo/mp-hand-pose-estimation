# 3D Hand Pose Estimation (Project 1) - Group KIdivergent
This repository presents an approach to automatically detect handpose and shape from RGB images with the help of a neural network. We enrich the provided freiHand dataset with online data augmentation and regress our model directly to the 21 3D joint positions.
  

## Branches:

- final submission branch: **resnet101_V1**
- each branch is accompanied with its own ReadMe to quickly outline what it's about

  
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

### Installing mano (for branches bou and bou_resnet50)
To download Mano-pytorch and Mano follow the installation procedure at https://github.com/hassony2/manopth
Furthermore, the environment variable `MANO` is needed: This should point to a directory which contains [this](https://github.com/hassony2/manopth/tree/master/mano) as well as a folder, called models, which

stores the file `MANO_RIGHT.pkl`, available at this [link](https://mano.is.tue.mpg.de/downloads)
  

