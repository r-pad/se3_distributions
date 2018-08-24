# generic_pose
Generic Pose Estimation Network

## Datasets
PoseImageDataset is the base dataset class for pose datasets. It is designed for use with pytorch's DataLoader. Child classes require getQuat, getImage, and \_\_len\_\_ to be implemented. The number of images being returned is controlled by loop_truth, which contains an array bools where cross model comparison is used if False and same model if True.

### Standard Datasets
NumpyImageDataset, found in generic_pose.datasets.numpy_dataset, is our standard pose dataset, used for both renders and data collects. This dataset takes in a text file containing a list of directories, the path to a directory, or a pkl of the data set. The directory structure should contain png images of the object as well as npy arrays of the objects orientation in the image. The directory structure should be of the form class/model/data.{png,npy}. Loading can be drastically speedup by saving the dataset to a pkl using the pkl_save_filename option.

```python
data_loader = DataLoader(NumpyImageDataset(data_folders='/path/to/data', img_size = (224, 224)),
                         num_workers=10, batch_size=32, shuffle=True)
# Loop over dataset
for j (imgs, trans, quats, mdl_names, mdl_fns) in enumerate(data_loader):
  pass

# Get a single batch of data (slower than for loop
imgs, trans, quats, mdl_names, mdl_fns = next(iter(data_loader))

# All data is preprocessed for pytoch networks
# imgs: list of images in loop. List of length 2, each element being a batch as 32 images. 
# trans: list of transforms (quaternions) from imgs[n] to imgs[n+1]
# quats: list of quaternions of imgs[n]
# mdl_names: list of names of model in imgs[n]
# mdl_fn: list of filepaths to mesh for imgs[n]  
```
### Benchmarking Datasets
LinemodDataset, found in generic_pose.datasets.benchmark_dataset, can be used to benchmark against the Linemod dataset. Images can be masked using the object point cloud and known transforms by setting the use_mask flag. Prerendering these masks can be done using render_scripts/render_masks.py. 

## Training
### Pose Distance Training

### Pose Step Training
### Pose Difference Training
### Tensorboard Logging w/ Pytorch

## Evaluation

## Pose Estimation Pipeline


