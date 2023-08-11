# 3D Deep Learning with Python

[Book](https://www.packtpub.com/product/3d-deep-learning-with-python/9781803247823)

## Setup environment

```
conda create -n dl3d python=3.7

conda activate dl3d

conda install pytorch torchvision torchaudio cudatoolkit-11.1 -c pytorch -c nvidia

conda install pytorch3d -c pytorch3d

conda install -c open3d-admin open3d

pip install -U scikit-learn scipy matplotlib

```

## Table Of Contents <a name="top"></a>

I. [Introduction - 3D Data processing](#c1)

1. [3D data representation](#c1.1)

    1.1. [Point cloud representation](#c1.1.1)
    
    1.2. [Mesh representation](#c1.1.2)
    
    1.3. [Voxel representation](#c1.1.3)

2. [3D data file formats](#c1.1.2)

    2.1. [Ply files](#c1.2.1)
    
    2.2. [Obj files](#c1.2.2)

3. [3D Coordinate Systems](#c1.1.3)
    
    3.1. [World coordinate system](#c1.3.1)
    
    3.2. [Normalized device coordinate (NDC)](#c1.3.2)

II. [Introduction - 3D Computer Vision and Geometry](#c2)

1. [Exploring the basic concepts of rendering, rasterization, and shading](#c2.1)

    1.1. [Barycentric coordinates](#c2.1.1)
    
    1.2. [Shading Models](#c2.1.2)

2. [Transformation and rotation](#c2.2)


III. [3D Deep Learning using PyTorch3D](#c3)
# I. Introduction - 3D Data processing <a name="c1"></a> 
 
## 1. 3D data representation <a name="c1.1"></a>

### 1.1. Point cloud representation <a name="c1.1.1"></a>

A 3D point cloud is a very straightforward representation of 3D objects, where each point cloud is just a collection of 3D points, and each 3D point is represented by one three-dimensional tuple (x, y, or z). The raw measurements of many depth cameras are usually 3D point clouds.

From a deep learning point of view, 3D point clouds are one of the unordered and irregular data types. Unlike regular images, where we can define neighboring pixels for each individual pixel, there are no clear and regular definitions for neighboring points for each point in a point cloud – that is, convolutions usually cannot be applied to point clouds.

Another issue for point clouds as training data for 3D deep learning is the heterogeneous data issue – that is, for one training dataset, different point clouds may contain different numbers of 3D points. One approach for avoiding such a heterogeneous data issue is forcing all the point clouds to have the same number of points. However, this may not be always possible – for example, the number of points returned by depth cameras may be different from frame to frame.

The heterogeneous data may create some difficulties for mini-batch gradient descent in training deep learning models. Most deep learning frameworks assume that each mini-batch contains training examples of the same size and dimensions. Such homogeneous data is preferred because it can be most efficiently processed by modern parallel processing hardware, such as GPUs. Handling heterogeneous mini-batches in an efficient way needs some additional work.

![Image](https://images.ctfassets.net/26961o1141cc/1ntbH068mqsmzD1v7P69hy/d6a023bdc9027478dca19a2b49c66b82/p6.png?w=1200&h=500&fm=webp&q=100)

### 1.2. Mesh representation <a name="c1.1.2"></a>

Meshes are another widely used 3D data representation. Like points in point clouds, each mesh contains a set of 3D points called vertices. In addition, each mesh also contains a set of polygons called faces, which are defined on vertices.

In most data-driven applications, meshes are a result of post-processing from raw measurements of depth cameras. Often, they are manually created during the process of 3D asset design. Compared to point clouds, meshes contain additional geometric information, encode topology, and have surface normal information. This additional information becomes especially useful in training learning models. For example, graph convolutional neural networks usually treat meshes as graphs and define convolutional operations using the vertex neighboring information.

Just like point clouds, meshes also have similar heterogeneous data issues.

![Image](https://upload.wikimedia.org/wikipedia/commons/f/fb/Dolphin_triangle_mesh.png)

### 1.3. Voxel representation <a name="c1.1.3"></a>

Another important 3D data representation is voxel representation. A voxel is the counterpart of a pixel in 3D computer vision. A pixel is defined by dividing a rectangle in 2D into smaller rectangles and each small rectangle is one pixel. Similarly, a voxel is defined by dividing a 3D cube into smaller-sized cubes and each cube is called one voxel.

Voxel representations usually use <b> Truncated Signed Distance Functions (TSDFs) </b> to represent 3D surfaces. A Signed Distance Function (SDF) can be defined at each voxel as the (signed) distance between the center of the voxel to the closest point on the surface. A positive sign in an SDF indicates that the voxel center is outside an object. The only difference between a TSDF and an SDF is that the values of a TSDF are truncated, such that the values of a TSDF always range from -1 to +1.

Unlike point clouds and meshes, voxel representation is ordered and regular. This property is like pixels in images and enables the use of convolutional filters in deep learning models. One potential disadvantage of voxel representation is that it usually requires more computer memory, but this can be reduced by using techniques such as hashing. Nevertheless, voxel representation is an important 3D data representation.

There are 3D data representations other than the ones mentioned here. For example, multi-view representations use multiple images taken from different viewpoints to represent a 3D scene. RGB-D representations use an additional depth channel to represent a 3D scene. However, in this book, we will not be diving too deep into these 3D representations. Now that we have learned the basics of 3D data representations, we will dive into a few commonly used file formats for point clouds and meshes.


![Image](https://static1.squarespace.com/static/5d7b6b83ace5390eff86b2ae/5fa172b055842d46da746f08/604f3b5bade8ee659ff4a633/1651616409490/3D_representations.jpg?format=1500w)


## 2. 3D data file formats <a name="c1.2"></a>

### 2.1. Ply files  <a name="c1.2.1"></a>

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

### 2.2. Obj files <a name="c1.2.2"></a>

Obj files are another commonly used file format for meshes. They are also text-based and human-readable. 

[Example of a Obj file](./Introduction/obj_io/README.md)

## 3. 3D Coordinate Systems <a name="c1.3"></a>

### 3.1. World coordinate system <a name="c1.3.1"></a>

The world coordinate system is the coordinate system that is used to represent the 3D world. It is usually defined by the 3D sensor that is used to capture the 3D data. For example, the world coordinate system of a depth camera is defined by the depth camera itself. The world coordinate system is usually defined as a right-handed coordinate system, where the x-axis points to the right, the y-axis points up, and the z-axis points forward.

![Image](/Course/Theory/3D-DeepLearning/image/WorldCoordinate.png)

### 3.2. Normalized device coordinate (NDC) <a name="c1.3.2"></a>

The normalized device coordinate (NDC) confines the volume that a camera can render. The x coordinate values in the NDC space range from -1 to +1, as do the y coordinate values. The z coordinate values range from znear to zfar, where znear is the nearest depth and zfar is the farthest depth. Any object out of this znear to zfar range would not be rendered by the camera.

Finally, the screen coordinate system is defined in terms of how the rendered images are shown on our screens. The coordinate system contains the x coordinate as the columns of the pixels, the y coordinate as the rows of the pixels, and the z coordinate corresponding to the depth of the object.

To render the 3D object correctly on our 2D screens, we need to switch between these coordinate systems.

![Image](/Course/Theory/3D-DeepLearning/image/NDC.png)


### 3.3. Camera models <a name="c1.3.3"></a>

![Image](/Course/Theory/3D-DeepLearning/image/cameraModel.png)

The orthographic cameras use orthographic projections to map objects in the 3D world to 2D images, while the perspective cameras use perspective projections to map objects in the 3D world to 2D images. The orthographic projections map objects to 2D images, disregarding the object depth. For example, just as shown in the figure, two objects with the same geometric size at different depths would be mapped to 2D images of the same size. On the other hand, in perspective projections, if an object moved far away from the camera, it would be mapped to a smaller size on the 2D images.

[Example of a Camera model](./Introduction/camera/README.md)

# II. Introduction - 3D Computer Vision and Geometry <a name="c2"></a> 

## 1. Exploring the basic concepts of rendering, rasterization, and shading <a name="c2.1"></a>

- <b>Rendering: </b> is a process that takes 3D data models of the world around our camera as input and output images. It is an approximation to the physical process where images are formed in our camera in the real world. Typically, the 3D data models are meshes. In this case, rendering is usually done using ray tracing:

![Image](/Course/Theory/3D-DeepLearning/image/rendering.png)

An example of ray tracing processing is shown in Figure above. In the example, the world model contains one 3D sphere, which is represented by a mesh model. To form the image of the 3D sphere, for each image pixel, we generate one ray, starting from the camera origin and going through the image pixel. If one ray intersects with one mesh face, then we know the mesh face can project its color to the image pixel. We also need to trace the depth of each intersection because a face with a smaller depth would occlude faces with larger depths.

Thus, the process of rendering can usually be divided into two stages – rasterization and shading.

- <b>Rasterization: </b> The ray tracing process is a typical rasterization process – that is, the process of finding relevant geometric objects for each image pixel.

- <b>Shading: </b> is the process of taking the outputs of the rasterization and computing the pixel value for each image pixel.

### 1.1. Barycentric coordinates <a name="c2.1.1"></a>

For each point coplanar with a mesh face, the coordinates of the point can always be written as a linear combination of the coordinates of the three vertices of the mesh face. For example, as shown in the following diagram, the point p can be written as $uA + vB + wC$ , where A, B, and C are the coordinates of the three vertices of the mesh face. Thus, we can represent each such point with the coefficients $u$, $v$, and $w$. This representation is called the <b>barycentric coordinates</b> of the point. For point lays within the mesh face triangle, $u + v + w = 1$ and all $u,v,w$ are positive numbers. Since barycentric coordinates define any point inside a face as a function of face vertices, we can use the same coefficients to interpolate other properties across the whole face as a function of the properties defined at the vertices of the face. For example, we can use it for shading as shown in Figure below:

![Image](/Course/Theory/3D-DeepLearning/image/barycentric.png)

### 1.2. Shading Models <a name="c2.1.2"></a>

#### 1.2.1. Light source models <a name="c2.1.2.1"></a>

Light propagation in the real world can be a sophisticated process. Several approximations of light sources are usually used in shading to reduce computational costs:

- The first assumption is ambient lighting, where we assume that there is some background light radiation after sufficient reflections, such that they usually come from all directions with almost the same amplitude at all image pixels.

- Another assumption that we usually use is that some light sources can be considered point light sources. A point light source radiates lights from one single point and the radiations at all directions have the same color and amplitude.

- A third assumption that we usually use is that some light sources can be modeled as directional light sources. In such a case, the light directions from the light source are identical at all the 3D spatial locations. Directional lighting is a good approximation model for cases where the light sources are far away from the rendered objects – for example, sunlight.

#### 1.2.2. Lambertian shading model <a name="c2.1.2.2"></a>

The first physical model that we will discuss is Lambert’s cosine law. Lambertian surfaces are types of objects that are not shiny at all, such as paper, unfinished wood, and unpolished stones:

![Image](/Course/Theory/3D-DeepLearning/image/lambertian.png)

Figure above shows an example of how lights diffuse on a Lambertian surface. One basic idea of the Lambertian cosine law is that for Lambertian surfaces, the amplitude of the reflected light does not depend on the viewer’s angle, but only depends on the angle $θ$ between the surface normal and the direction of the incident light. More precisely, the intensity of the reflected light $c$ is as follows:

$c = c_{r} c_{l} cos(\theta)$

where $c_{r}$ is the amplitude of the reflected light, $c_{l}$ is the amplitude of the incident light, and $θ$ is the angle between the surface normal and the direction of the incident ligh

If we further consider the ambient light, the amplitude of the reflected light is as follows:

$c = c_{r} (c_{a} + c_{l} cos(\theta))$

where $c_{a}$ is the amplitude of the ambient light.

#### 1.2.3. Phong shading model <a name="c2.1.2.3"></a>

For shiny surfaces, such as polished tile floors and glossy paint, the reflected light also contains a highlight component. The Phong lighting model is a frequently used model for these glossy components:

![Image](/Course/Theory/3D-DeepLearning/image/phong.png)

An example of the Phong lighting model is shown in Figure above. One basic principle of the Phong lighting model is that the shiny light component should be strongest in the direction of reflection of the incoming light. The component would become weaker as the angle $c$ between the direction of
reflection and the viewing angle becomes larger.

More precisely, the amplitude of the shiny light component $c$ is equal to the following:

$c = c_{r} c_{l} c_{p} [cos(σ)]^p$

Here, the exponent $p$ is a parameter of the model for controlling the speed at which the shiny components attenuate when the viewing angle is away from the direction of reflection.

Finally, if we consider all three major components – ambient lighting, diffusion, and highlights – the final equation for the amplitude of light is as follows:

$c = c_{r} (c_{a} + c_{l} cos(\theta) + c_{l} c_{p} [cos(σ)]^p)$

Note that the preceding equation applies to each color component. In other words, we will have one of these equations for each color channel (red, green, and blue) with a distinct set of $c_{r}$, $c_{l}$, $c_{a}$ values.

[Example for 3D rendering](./3DRendering/)

## 2. Transformation and rotation <a name="c2.2"></a>

SO(3) denotes the special orthogonal group in 3D and SE(3) denotes the special Euclidean group in 3D. Informally speaking, SO(3) denotes the set of all the rotation transformations and SE(3) denotes the set of all the rigid transformations in 3D.

[Pytoech3D example](./3DRendering/)

# III. 3D Deep Learning using PyTorch3D <a name="c3"></a>

## 1. Fitting Deformable Mesh Models to Raw Point Clouds <a name="c3.1"></a>

### 1.1. Fitting meshes to point clouds – the problem <a name="c3.1.1"></a>

Real-world depth cameras, such as LiDAR, time-of-flight cameras, and stereo vision cameras, usually output either depth images or point clouds. For example, in the case of time-of-flight cameras, a modulated light ray is projected from the camera to the world, and the depth at each pixel is measured from the phase of the reflected light rays received at the pixel. Thus, at each pixel, we can usually get one depth measurement and one reflected light amplitude measurement. However, other than the sampled depth information, we usually do not have direct measurements of the surfaces. For example, we cannot measure the smoothness or norm of the surface directly.

Similarly, in the case of stereo vision cameras, at each time slot, the camera can take two RGB images from the camera pair at roughly the same time. The camera then estimates the depth by finding the pixel correspondences between the two images. The output is thus a depth estimation at each pixel. Again, the camera cannot give us any direct measurements of surfaces.

However, in many real-world applications, surface information is sought. For example, in robotic picking tasks, usually, we need to find regions on an object such that the robotic hands can grasp firmly. In such a scenario, it is usually desirable that the regions are large in size and reasonably flat.

There are many other scenarios in which we want to fit a (deformable) mesh model to a point cloud. For example, there are some machine vision applications where we have the mesh model for an industrial part and the point cloud measurement from the depth camera has an unknown orientation and pose. In this case, finding a fitting of the mesh model to the point cloud would recover the unknown object pose.

For another example, in human face tracking, sometimes, we want to fit a deformable face mesh model to point cloud measurements, such that we can recover the identity of the human being and/or facial expressions.

<b>Loss functions</b> are central concepts in almost all optimizations. Essentially, to fit a point cloud, we need to design a loss function, such that when the loss function is minimized, the mesh as the optimization variable fits to the point cloud.

Actually, selecting the right loss function is usually a critical design decision in many real-world projects. Different choices of loss function usually result in significantly different system performance. The requirements for a loss function usually include at least the following properties:

• The loss function needs to have desirable numerical properties, such as smooth, convex, without the issue of vanishing gradients, and so on

• The loss function (and its gradients) can be easily computed; for example, they can be efficiently computed on GPUs

• The loss function is a good measurement of model fitting; that is, minimizing the loss function results in a satisfactory mesh model fitting for the input point clouds

Other than one primary loss function in such model fitting optimization problems, we usually also need to have other loss functions for regularizing the model fitting. For example, if we have some prior knowledge that the surfaces should be smooth, then we usually need to introduce an additional regularization loss function, such that not-smooth meshes would be penalized more.

### 1.2. Formulating a deformable mesh fitting problem into an optimization problem <a name="c3.1.2"></a>

In this section, we are going to talk about how to formulate the mesh fitting problem into an optimization problem. One key observation here is that object surfaces such as pedestrians can always be continuously deformed into a sphere. Thus, the approach we are going to take will start from the surface of a sphere and deform the surface to minimize a cost function.

The cost function should be chosen such that it is a good measurement of how similar the point cloud is to the mesh. Here, we choose the major cost function to be the Chamfer set distance. The Chamfer distance is defined between two sets of points as follows:

![Image](/Course/Theory/3D-DeepLearning/image/ChamferDistance.png)

The Chamfer distance is symmetric and is a sum of two terms. In the first term, for each point x in the first point cloud, the closest point y in the other point cloud is found. For each such pair x and y, their distance is obtained and the distances for all the pairs are summed up. Similarly, in the second term, for each y in the second point cloud, one x is found and the distances between such x and y pairs are summed up.

The Chamfer distance is a good measurement of how similar two point clouds are. If the two point clouds are the same, then the Chamfer distance is zero. If the two point clouds are different, then the Chamfer distance is positive.

### 1.3. Loss functions for regularization <a name="c3.1.3"></a>

In the previous section, we successfully formulated the deformable mesh fitting problem into an optimization problem. However, the approach of directly optimizing this primary loss function can be problematic. The issues lie in that there may exist multiple mesh models that can be good fits to the same point cloud. These mesh models that are good fits may include some mesh models that are far away from smooth meshes.

On the other hand, we usually have prior knowledge about pedestrians. For example, the surfaces of pedestrians are usually smooth, the surface norms are smooth also. Thus, even if a non-smooth mesh is close to the input point cloud in terms of Chamfer distance, we know with a certain level of confidence that it is far away from the ground truth.

Machine learning literature has provided solutions for excluding such undesirable non-smooth solutions for several decades. The solution is called <b>regularization</b>. Essentially, the loss we want to optimize is chosen to be a sum of multiple loss functions. Certainly, the first term of the sum will be the primary Chamfer distance. The other terms are for penalizing surface non-smoothness and norm non-smoothness.

#### 1.3.1. Mesh Laplacian smoothing loss <a name="c3.1.3.1"></a>

The mesh Laplacian is a discrete version of the well-known Laplace-Beltrami operator. One version (usually called uniform Laplacian) is as follows:

![Image](/Course/Theory/3D-DeepLearning/image/Laplacian.png)

In the preceding definition, the Laplacian at the i-th vertex is just a sum of differences, where each difference is between the coordinates of the current vertex and those of a neighboring vertex.

The Laplacian is a measurement for smoothness. If the i-th vertex and its neighbors lie all within one plane, then the Laplacian should be zero. Here, we are using a uniform version of the Laplacian, where the contribution to the sum from each neighbor is equally weighted. There are more complicated versions of Laplacians, where the preceding contributions are weighted according to various schemes.

#### 1.3.2. Mesh normal consistency loss <a name="c3.1.3.2"></a>

The mesh normal consistency loss is a loss function for penalizing the distances between adjacent normal vectors on the mesh.

#### 1.3.3. Mesh edge loss <a name="c3.1.3.3"></a>

Mesh edge loss is for penalizing long edges in meshes. For example, in the mesh model fitting problem we consider in this chapter, we want to eventually obtain a solution, such that the obtained mesh model fits the input point cloud uniformly. In other words, each local region of the point cloud is covered by small triangles of the mesh. Otherwise, the mesh model cannot capture the fine details of slowly varying surfaces, meaning the model may not be that accurate or trustworthy.

The aforementioned problem can be easily avoided by including the mesh edge loss in the objective function. The mesh edge loss is essentially a sum of all the edge lengths in the mesh.

[Implementing the mesh fitting with PyTorch3D](./meshesFitting/)

## 2. Object Pose Detection and Tracking by Differentiable Rendering <a name="c3.2"></a>

### 2.1. Why we want to have differentiable rendering <a name="c3.2.1"></a>

The physical process of image formation is a mapping from 3D models to 2D images. As shown in the example in Figure below, depending on the positions of the red and blue spheres in 3D (two possible configurations are shown on the left-hand side), we may get different 2D images (the images corresponding to the two configurations are shown on the right-hand side).

![Image](/Course/Theory/3D-DeepLearning/image/mapping3Dto2D.png)

Many 3D computer vision problems are a reversal of image formation. In these problems, we are usually given 2D images and need to estimate the 3D models from the 2D images. For example, in Figure below, we are given the 2D image shown on the right-hand side and the question is, which 3D model is the one that corresponds to the observed image?

![Image](/Course/Theory/3D-DeepLearning/image/mapping2Dto3D.png)

According to some ideas that were first discussed in the computer vision community decades ago, we can formulate the problem as an optimization problem. In this case, the optimization variables here are the position of two 3D spheres. We want to optimize the two centers, such that the rendered images are like the preceding 2D observed image. To measure similarity precisely, we need to use a cost function – for example, we can use pixel-wise mean-square errors. We then need to compute a gradient from the cost function to the two centers of spheres, so that we can minimize the cost function iteratively by going toward the gradient descent direction.

However, we can calculate a gradient from the cost function to the optimization variables only under the condition that the mapping from the optimization variables to the cost functions is differentiable, which implies that the rendering process is also differentiable.

### 2.2. How to make rendering differentiable <a name="c3.2.2"></a>

Rendering is an imitation of the physical process of image formation. This physical process of image formation itself is differentiable in many cases. Suppose that the surface is normal and the material properties of the object are all smooth. Then, the pixel color in the example is a differentiable function of the positions of the spheres.

However, there are cases where the pixel color is not a smooth function of the position. This can happen at the occlusion boundaries, for example. This is shown in Figure 4.3, where the blue sphere is at a location that would occlude the red sphere at that view if the blue sphere moved up a little bit. The pixel moved at that view is thus not a differentiable function of the sphere center locations.

![Image](/Course/Theory/3D-DeepLearning/image/occlusion.png)

When we use conventional rendering algorithms, information about local gradients is lost due to discretization. As we discussed in the previous section, rasterization is a step of rendering where for each pixel on the imaging plane, we find the most relevant mesh face (or decide that no relevant mesh face can be found).

In conventional rasterization, for each pixel, we generate a ray from the camera center going through the pixel on the imaging plane. We will find all the mesh faces that intersect with this ray. In the conventional approach, the rasterizer will only return the mesh face that is nearest to the camera. The returned mesh face will then be passed to the shader, which is the next step of the rendering pipeline. The shader will then be applied to one of the shading algorithms (such as the Lambertian model or Phong model) to determine the pixel color. This step of choosing the mesh to render is a non-differentiable process, since it is mathematically modeled as a step function.

There has been a large body of literature in the computer vision community on how to make rendering differentiable.
The differentiable rendering implemented in the PyTorch3D library mainly used the approach in <b>Soft Rasterizer</b> by Liu, Li, Chen, and Li (arXiv:1904.01786).

The main idea of differentiable rendering is illustrated in Figure below. In the rasterization step, instead of returning only one relevant mesh face, we will find all the mesh faces, such that the distance of the mesh face to the ray is within a certain threshold.

![Image](/Course/Theory/3D-DeepLearning/image/differentableRendering.png)

<b>What problems can be solved by using differentiable rendering?</b>

Differentiable rendering is a technique in that we can formulate the estimation problems in 3D computer vision into optimization problems. It can be applied to a wide range of problems. More interestingly, one exciting recent trend is to combine differentiable rendering with deep learning. Usually, differentiable rendering is used as the generator part of the deep learning models. The whole pipeline can thus be trained end to end.


### 2.3. The object pose estimation problem <a name="c3.2.3"></a>

In this section, we are going to show a concrete example of using differentiable rendering for 3D computer vision problems. The problem is object pose estimation from one single observed image. In addition, we assume that we have the 3D mesh model of the object.

For example, we assume we have the 3D mesh model for a toy cow, as shown in Figure 2.3.1. Now, suppose we have taken one image of the toy cow (Figure 2.3.2). The problem is then to estimate the orientation and location of the toy cow at the moments when these images are taken.

![Image](/Course/Theory/3D-DeepLearning/image/cow1.png)

![Image](/Course/Theory/3D-DeepLearning/image/cow2.png)

Because it is cumbersome to rotate and move the meshes, we choose instead to fix the orientations and locations of the meshes and optimize the orientations and locations of the cameras. By assuming that the camera orientations are always pointing toward the meshes, we can further simplify the problem, such that all we need to optimize is the camera locations.

Thus, we formulate our optimization problem, such that the optimization variables will be the camera locations. By using differentiable rendering, we can render RGB images and silhouette images for the two meshes. The rendered images are compared with the observed images and, thus, loss functions between the rendered images and observed images can be calculated. Here, we use mean-square errors as the loss function. Because everything is differentiable, we can then compute gradients from the loss functions to the optimization variables. Gradient descent algorithms can then be used to find the best camera positions, such that the rendered images are matched to the observed images.

[Code Implementation](./objectPoseEstimation/)












