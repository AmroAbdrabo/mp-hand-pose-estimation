## 3D Hand Pose Estimation (Project 1) - Group KIdivergent

### Installing mano
To download Mano-pytorch and Mano follow the installtion procedure at https://github.com/hassony2/manopth

### Installing requirements
There are additional requirement that can be pip-installed in requirements.txt

### Downloading 128x128 background images

To get the backgrounds used for data augmentation, download this [dataset](http://chaladze.com/l5/img/Linnaeus%205%20128X128.rar)  of 128x128 images and put all the images in folder _other_ to a directory.

### Environment variables

Make sure you have these environments variables:

- `MANO` : This should point to a directory which contains [this](https://github.com/hassony2/manopth/tree/master/mano)  as well as a folder, called models, which
stores the file `MANO_RIGHT.pkl`, available in this [link](https://mano.is.tue.mpg.de/) 
- `MP_BACKGROUNDS` : This should point to the direcotry which stores all the 128x128 background images
- `MP_EXPERIMENTS`: This should point to the directory where you will output experiments runs (folders which contain the saved state of the model and optimizer)
- `MP_DATA`: This should point to the directory `freihand_dataset_MP` folder (located in /cluster/project/infk/hilliges/lectures/mp21/project1/ on the cluster). 