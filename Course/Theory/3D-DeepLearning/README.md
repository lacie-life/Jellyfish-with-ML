# 3D Deep Learning with Python

[Book](https://www.packtpub.com/product/3d-deep-learning-with-python/9781803247823)

## Setup environment

```
conda create -n dl3d python=3.7

conda activate dl3d

conda install pytorch torchvision torchaudio cudatoolkit-11.1 -c pytorch -c nvidia

conda install pytorch3d -c pytorch3d
```

## Table Of Contents <a name="top"></a>

I. [Introduction - 3D Data processing](#intro1)

1. [3D data representation](#1)

    1.1. [Point cloud representation](#1.1)
    
    1.2. [Mesh representation](#1.2)
    
    1.3. [Voxel representation](#1.3)

2. [3D data file formats](#1.2)

    2.1. [Ply files](#2.1)
    
    2.2. [Obj files](#2.2)

3. [3D Coordinate Systems](#1.3)
    
    3.1. [World coordinate system](#3.1)
    
    3.2. [Normalized device coordinate (NDC)](#3.2)

II. [Introduction - 3D Computer Vision and Geometry](#intro2)

# I. Introduction - 3D Data processing <a name="intro1"></a> 
 
## 1. 3D data representation <a name="1"></a>

### 1.1. Point cloud representation <a name="1.1"></a>

A 3D point cloud is a very straightforward representation of 3D objects, where each point cloud is just a collection of 3D points, and each 3D point is represented by one three-dimensional tuple (x, y, or z). The raw measurements of many depth cameras are usually 3D point clouds.

From a deep learning point of view, 3D point clouds are one of the unordered and irregular data types. Unlike regular images, where we can define neighboring pixels for each individual pixel, there are no clear and regular definitions for neighboring points for each point in a point cloud – that is, convolutions usually cannot be applied to point clouds.

Another issue for point clouds as training data for 3D deep learning is the heterogeneous data issue – that is, for one training dataset, different point clouds may contain different numbers of 3D points. One approach for avoiding such a heterogeneous data issue is forcing all the point clouds to have the same number of points. However, this may not be always possible – for example, the number of points returned by depth cameras may be different from frame to frame.

The heterogeneous data may create some difficulties for mini-batch gradient descent in training deep learning models. Most deep learning frameworks assume that each mini-batch contains training examples of the same size and dimensions. Such homogeneous data is preferred because it can be most efficiently processed by modern parallel processing hardware, such as GPUs. Handling heterogeneous mini-batches in an efficient way needs some additional work.

![Image](https://images.ctfassets.net/26961o1141cc/1ntbH068mqsmzD1v7P69hy/d6a023bdc9027478dca19a2b49c66b82/p6.png?w=1200&h=500&fm=webp&q=100)

### 1.2. Mesh representation <a name="1.2"></a>

Meshes are another widely used 3D data representation. Like points in point clouds, each mesh contains a set of 3D points called vertices. In addition, each mesh also contains a set of polygons called faces, which are defined on vertices.

In most data-driven applications, meshes are a result of post-processing from raw measurements of depth cameras. Often, they are manually created during the process of 3D asset design. Compared to point clouds, meshes contain additional geometric information, encode topology, and have surface normal information. This additional information becomes especially useful in training learning models. For example, graph convolutional neural networks usually treat meshes as graphs and define convolutional operations using the vertex neighboring information.

Just like point clouds, meshes also have similar heterogeneous data issues.

![Image](https://www.researchgate.net/publication/322096576/figure/fig2/AS:631626539229214@1527602910310/3D-mesh-triangles-with-different-resolution-3D-Modelling-for-programmers-Available-at.png)

### 1.3. Voxel representation <a name="1.3"></a>

Another important 3D data representation is voxel representation. A voxel is the counterpart of a pixel in 3D computer vision. A pixel is defined by dividing a rectangle in 2D into smaller rectangles and each small rectangle is one pixel. Similarly, a voxel is defined by dividing a 3D cube into smaller-sized cubes and each cube is called one voxel.

Voxel representations usually use <b> Truncated Signed Distance Functions (TSDFs) </b> to represent 3D surfaces. A Signed Distance Function (SDF) can be defined at each voxel as the (signed) distance between the center of the voxel to the closest point on the surface. A positive sign in an SDF indicates that the voxel center is outside an object. The only difference between a TSDF and an SDF is that the values of a TSDF are truncated, such that the values of a TSDF always range from -1 to +1.

Unlike point clouds and meshes, voxel representation is ordered and regular. This property is like pixels in images and enables the use of convolutional filters in deep learning models. One potential disadvantage of voxel representation is that it usually requires more computer memory, but this can be reduced by using techniques such as hashing. Nevertheless, voxel representation is an important 3D data representation.

There are 3D data representations other than the ones mentioned here. For example, multi-view representations use multiple images taken from different viewpoints to represent a 3D scene. RGB-D representations use an additional depth channel to represent a 3D scene. However, in this book, we will not be diving too deep into these 3D representations. Now that we have learned the basics of 3D data representations, we will dive into a few commonly used file formats for point clouds and meshes.


![Image](https://static1.squarespace.com/static/5d7b6b83ace5390eff86b2ae/5fa172b055842d46da746f08/604f3b5bade8ee659ff4a633/1651616409490/3D_representations.jpg?format=1500w)


## 2. 3D data file formats <a name="2"></a>

### 2.1. Ply files  <a name="2.1"></a>

The PLY file format is one of the most commonly used file formats for point clouds and meshes. It is a simple file format that can be easily parsed by most programming languages. The PLY file format is a text-based file format, which means that it is human-readable. The following is an example of a PLY file:

```
ply
format ascii 1.0
comment created for the book 3D Deep Learning with Python
element vertex 8
property float32 x
property float32 y
property float32 z
element face 12
property list uint8 int32 vertex_indices
end_header
-1 -1 -1
1 -1 -1
1 1 -1
-1 1 -1
-1 -1 1
1 -1 1
1 1 1
-1 1 1
3 0 1 2
3 5 4 7
3 6 2 1
3 3 7 4
3 7 3 2
3 5 1 0
3 0 2 3
3 5 7 6
3 6 1 5
3 3 4 0
3 7 2 6
3 5 0 4
```

[Example of a PLY file](./Introduction/ply_io/README.md)

### 2.2. Obj files <a name="2.2"></a>

Obj files are another commonly used file format for meshes. They are also text-based and human-readable. 

[Example of a Obj file](./Introduction/obj_io/README.md)

## 3. 3D Coordinate Systems <a name="3"></a>

### 3.1. World coordinate system <a name="3.1"></a>

The world coordinate system is the coordinate system that is used to represent the 3D world. It is usually defined by the 3D sensor that is used to capture the 3D data. For example, the world coordinate system of a depth camera is defined by the depth camera itself. The world coordinate system is usually defined as a right-handed coordinate system, where the x-axis points to the right, the y-axis points up, and the z-axis points forward.

![Image](/Course/Theory/3D-DeepLearning/image/WorldCoordinate.png)

### 3.2. Normalized device coordinate (NDC) <a name="3.2"></a>

The normalized device coordinate (NDC) confines the volume that a camera can render. The x coordinate values in the NDC space range from -1 to +1, as do the y coordinate values. The z coordinate values range from znear to zfar, where znear is the nearest depth and zfar is the farthest depth. Any object out of this znear to zfar range would not be rendered by the camera.

Finally, the screen coordinate system is defined in terms of how the rendered images are shown on our screens. The coordinate system contains the x coordinate as the columns of the pixels, the y coordinate as the rows of the pixels, and the z coordinate corresponding to the depth of the object.

To render the 3D object correctly on our 2D screens, we need to switch between these coordinate systems.

![Image](/Course/Theory/3D-DeepLearning/image/NDC.png)


### 3.3. Camera models <a name="3.3"></a>

![Image](/Course/Theory/3D-DeepLearning/image/cameraModel.png)

The orthographic cameras use orthographic projections to map objects in the 3D world to 2D images, while the perspective cameras use perspective projections to map objects in the 3D world to 2D images. The orthographic projections map objects to 2D images, disregarding the object depth. For example, just as shown in the figure, two objects with the same geometric size at different depths would be mapped to 2D images of the same size. On the other hand, in perspective projections, if an object moved far away from the camera, it would be mapped to a smaller size on the 2D images.

[Example of a Camera model](./Introduction/camera/README.md)

# II. Introduction - 3D Computer Vision and Geometry <a name="intro2"></a> 
