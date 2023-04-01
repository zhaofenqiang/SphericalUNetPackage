# [Spherical U-Net package](https://pypi.org/project/sphericalunet/)
Python-based spherical cortical surface processing tools, including spherical resampling, interpolation, parcellation, registration, atlas construction, etc. It provides fast and accurate cortical surface-based data analysis using deep learning techniques.

## Install

It can be installed from PyPI using:

```
pip install sphericalunet
```

Or download packed ready-to-use tools from [Nitrc](https://www.nitrc.org/projects/infantsurfparc)

## Main tools
[**I/O vtk file**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/main/sphericalunet/utils/vtk.py). Python function for reading and writing .vtk surface file. Example code:
```
from sphericalunet.utils.vtk import read_vtk, write_vtk

surf = read_vtk(file_name)
# some operations to the surface 
write_vtk(surf, new_file_name)
```
For matlab users, please refer to this [issue](https://github.com/zhaofenqiang/Spherical_U-Net/issues/3#issuecomment-763334969) and this [repository](https://github.com/Zhengwang-Wu/CorticalSurfaceMetric).

[**Layers**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/main/sphericalunet/models/layers.py) provide basic spherical convolution, pooling, upsampling layers for constructing spherical convolutional neural networks.

[**Models**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/main/sphericalunet/models/models.py) provide some baseline spherical convolutional neural networks, e.g., [Spherical U-Net](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/models/models.py#L266), [Spherical VGG](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/models/models.py#L420), [Spherical ResNet](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/b17add7b1259db187bbf9321cba2ec34e5e4be8e/sphericalunet/models/models.py#L494), Spherical CycleGAN, etc.

[**Resample feature**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/utils/interp_numpy.py#L316) on spherical surface to standard icosahedron subdivision spheres. Example code:
```
from sphericalunet.utils.interp_numpy import resampleSphereSurf
from sphericalunet.utils.vtk import read_vtk, write_vtk

template_163842 = read_vtk('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk')
neigh_orders_163842 = get_neighs_order(163842)

data = read_vtk(file)
resampled_feat = resampleSphereSurf(data['vertices'], template_163842['vertices'], 
			             np.concatenate((data['sulc'][:,np.newaxis], data['curv'][:,np.newaxis]), axis=1),
			             neigh_orders=neigh_orders_163842)
surf = {'vertices': template_163842['vertices'], 
        'faces': template_163842['faces'],
        'sulc': resampled_feat[:,0],
        'curv': resampled_feat[:,1]}
    
write_vtk(surf, file.replace('.vtk', '.resample.vtk'))
```
Note if you want to run it on GPU, change `interp_numpy` to `interp_torch`.

[**Resample label**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/utils/vtk.py#L99) on spherical surface to standard icosahedron subdivision spheres. Example code:
```
from sphericalunet.utils.vtk import read_vtk, write_vtk, resample_label

template_163842 = read_vtk('/media/ychenp/DATA/unc/Data/Template/sphere_163842.vtk')

surf = read_vtk(file)
resampled_par = resample_label(surf['vertices'], template_163842['vertices'], surf['par_fs_vec'])
```

[**Smooth feature**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/utils/vtk.py#L153) on spherical surface.

[**Cortical surface parcellation**](https://github.com/zhaofenqiang/Spherical_U-Net) with trained models based on this package.

[**Deformable cortical surface registration**](https://github.com/zhaofenqiang/spherical-registration) with trained models based on this package.

[**Rigid cortical surface registration**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/99963658ab4690c198b337aad99a099791753902/sphericalunet/utils/initial_rigid_align.py#L35). An example code can be found [here](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/main/example/initialRigidAlignUsingSearch_longleaf.py).

[**Chcek folded triangles**](https://github.com/zhaofenqiang/SphericalUNetPackage/blob/b17add7b1259db187bbf9321cba2ec34e5e4be8e/sphericalunet/utils/utils.py#L496), and correct them (not implemented yet...).



## Papers

If you use this code, please cite:

Fenqiang Zhao, et.al. [Spherical U-Net on Cortical Surfaces: Methods and Applications](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_67). Information Processing in Medical Imaging (IPMI), 2019.

Fenqiang Zhao, et.al. [Spherical Deformable U-Net: Application to Cortical Surface Parcellation and Development Prediction](https://ieeexplore.ieee.org/document/9316936). IEEE Transactions on Medical Imaging, 2021.

Fenqiang Zhao, et.al. [S3Reg: Superfast Spherical Surface Registration Based on Deep Learning](https://ieeexplore.ieee.org/document/9389746). IEEE Transactions on Medical Imaging, 2021.

