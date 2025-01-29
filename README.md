Training Dataset Generation Using Gaussian Splatting
=====================================================

## Prerequisites

To install PyCuda, run the following command in the command prompt (assuming you are in the project root directory):
```bash
git clone https://github.com/inducer/pycuda
cd pycuda
git submodule update --init
export PATH=$PATH:/usr/local/cuda/bin
./configure.py --cuda-enable-gl
python setup.py install
cd ..
```

## Running the Scripts

For creating the training dataset use the following command:

```bash
python.exe .\batch_render.py --conf <path_to_config_file>
```
Running this script requires access to the point cloud data (in the form of NetCDF files) and training camera views (as RT transformation matrices). Refer to [NARVis](https://github.com/VALIXR-Neural-Rendering/narvis) codebase to access the Storms dataset and also check out the format of the camera view matrices to be used with this codebase. Use the `--help` option for more information on supported options for the scripts.

For quick viewing and experimenting with the splat features of the renderings we provide a GSplat viewer. To run this viewer use: the following command:

```bash
python.exe .\viewer.py --conf <path_to_config_file>
```

Use mouse left button to explore the scene and use scroll to zoom in/out.


## Acknowledgements

- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) - Original Python module (over native CUDA/C++ implementation) of fully differentiable Gaussian Splatting renderer.
- [diff-gaussian-rasterization-depth-acc](https://github.com/robot0321/diff-gaussian-rasterization-depth-acc) - Augments the `diff-gaussian-rasterization` codebase with mean depth and accuracy maps. This is used to render both RGB and depth maps of the scene in our codebase