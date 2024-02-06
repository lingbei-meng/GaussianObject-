# GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting

### [Project Page](https://gaussianobject.github.io/) | [Paper](https://raw.githubusercontent.com/GaussianObject/gaussianobject.github.io/main/assets/paper.pdf) | [Paper (Compressed)](https://raw.githubusercontent.com/GaussianObject/gaussianobject.github.io/main/assets/paper_compressed.pdf)

**The code will be released in a few days! Please stay tuned!**

We propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting, that achieves high rendering quality with only **4 input images**.

https://github.com/GaussianObject/GaussianObject/assets/158549428/70ae2443-7a6e-4352-abf4-d3abf79779a3

We first introduce techniques of visual hull and floater elimination which explicitly inject structure priors into the initial optimization process for helping build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. Our GaussianObject achives strong reconstruction results from only 4 views and significantly outperforms previous state-of-the-art methods.

![pipeline](assets/pipe.png)

- We initialize 3D Gaussians by constructing a visual hull with camera parameters and masked images, optimizing them with the $\mathcal{L}_{\text{gs}}$ and refining through floater elimination.
- We use a novel `leave-one-out' strategy and add 3D noise to Gaussians to generate corrupted Gaussian renderings. These renderings, paired with their corresponding reference images, facilitate the training of the Gaussian repair model employing $\mathcal{L}_{\text{tune}}$.
- Once trained, the Gaussian repair model is frozen and used to correct views that need to be rectified. These views are identified through distance-aware sampling. The repaired images and reference images are used to further optimize 3D Gaussians with $\mathcal{L}_{\text{rep}}$ and $\mathcal{L}_{\text{gs}}$.
