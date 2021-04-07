# Learning Orientation Distributions for Object Pose Estimation
Created by Brian Okorn at [R-PaD lab](https://r-pad.github.io/) at the Carnegie Mellon Robotics Institute. 

## Overview
We introduce two learned methods for estimating a distribution over an object's orientation. Our methods take into account both the inaccuracies in the pose estimation as well as the object symmetries. Our first method, which regresses from deep learned features to an isotropic Bingham distribution, gives the best performance for orientation distribution estimation for non-symmetric objects. Our second method learns to compare deep features and generates a non-parameteric histogram distribution. This method gives the best performance on objects with unknown symmetries, accurately modeling both symmetric and non-symmetric objects, without any requirement of symmetry annotation. [Project](https://bokorn.github.io/orientation-distributions/), [arXiv] (https://arxiv.org/abs/2007.01418).

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

## Installation

quat_math
object_pose_utils
pybingham

This code can be run as a stand alone library by pip installing the root directory. 
```
pip install .
```
## Pretrained Weights and Feature Grid

https://drive.google.com/drive/folders/1n6Ya0YfkGaXuWVEYlWvMs9coibZv5vKz?usp=sharing

## Usage
See [notebooks/](notebooks/) for interactive examples of using our models and datasets.

## Training 
See  [training_scripts/](training_scripts/) for example scripts for training orientation distribution networks.

## Renderer
