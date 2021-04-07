# Learning Orientation Distributions for Object Pose Estimation
Created by Brian Okorn at [R-PaD lab](https://r-pad.github.io/) at the Carnegie Mellon Robotics Institute. 

## Overview
We introduce two learned methods for estimating a distribution over an object's orientation. Our methods take into account both the inaccuracies in the pose estimation as well as the object symmetries. Our first method, which regresses from deep learned features to an isotropic Bingham distribution, gives the best performance for orientation distribution estimation for non-symmetric objects. Our second method learns to compare deep features and generates a non-parameteric histogram distribution. This method gives the best performance on objects with unknown symmetries, accurately modeling both symmetric and non-symmetric objects, without any requirement of symmetry annotation. [Project](https://bokorn.github.io/orientation-distributions/), [arXiv](https://arxiv.org/abs/2007.01418).

## Citation
If you find our work useful, please consider citing:
```
@inproceedings{okorn2020learning,
    Author = {Okorn, Brian and Xu, Mengyun and Hebert, Martial and Held, David },
    Title = {Learning Orientation Distributions for Object Pose Estimation},
    Journal   = {International Conference on Intelligent Robots and Systems (IROS)},
    Year = {2020}
}
```

## Requirements

### [quat_math](https://github.com/r-pad/quat_math)
Library of orientation helper functions as well as a wrapper to Christoph Gohlke's original transform library.
To install, pip install in the root directory
```
pip install .
```
### [object_pose_utils](https://github.com/r-pad/object_pose_utils)
Utility functions, datasets, and wrapper to Julian Straub's 4D spherical descritization [code](https://github.com/jstraub/dpOptTrans)
To install, pip install in the root directory
```
pip install .
```
### [pybingham](https://github.com/r-pad/bingham)
To use the Bingham distribution code, install our our python wrapper to the Bingham Statistics Library from our fork. Follow the python install instructions [here](https://github.com/r-pad/bingham/blob/master/python/INSTALL).

### [DenseFusion](https://github.com/r-pad/DenseFusion)
To install Dense Fusion as a stand alone library, generate new features, and to have access to the dropout version described in the paper, install our fork. To install, pip install in the root directory
```
pip install .
```
### [PoseCNN](https://github.com/r-pad/PoseCNN)
To generate features for PoseCNN, use our fork and this [tool](https://github.com/r-pad/PoseCNN/blob/master/tools/calc_features_aug.py) or the [notebook](https://github.com/r-pad/PoseCNN/blob/master/PoseCNN_Dataset_and_Featureizer.ipynb).

## Installation
This code should be installed as a stand alone library using pip in the root directory. 
```
pip install .
```
## Datasets
Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/)

## Pretrained Weights and Feature Grid
Download our pretrained models and generated feature grids [here](https://drive.google.com/drive/folders/1n6Ya0YfkGaXuWVEYlWvMs9coibZv5vKz?usp=sharing).

## Usage
See [notebooks/](notebooks/) for interactive examples of using our models and datasets.

## Training 
See  [training_scripts/](training_scripts/) for example scripts for training orientation distribution networks.

## Renderer
To render models for grids, you can use our stand alone Blender [renderer](https://github.com/r-pad/model_renderer) or pyrender.
