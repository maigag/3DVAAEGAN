# 3DVAAEGAN

We introduce 3DVAAEGAN, a 3D fully-convolutional variational adversarial auto encoder generative adversarial network 
that takes an incomplete 3D scan of a scene as input and performs scene and semantic completion. 
The main contribution of our model is to perform reconstruction at a high resolution, in a realistic manner, and on 
arbitrary scene sizes, combining the power of GANs and VAEs 
and helping the latter modeling the latent space with the support of an adversarial loss.

<img src="./img/RESULT.png" width="500">

# Introduction

In a time where augmented and virtual reality applications are ever-increasing and where RGB-D cameras are everywhere, optimal 3D reconstruction of indoor environments gained importance and momentum.
State-of-the-art scanning techniques, though, do not allow satisfying scenes recreation and are always leading to the presence of holes in the reassembling. The latters often conduce to major issues when it comes to applications such as scene editing or 3D printing.The automatic completion of surfaces and semantics in 3D is a smart way to deceive this problem, but traditional methods such as Poisson Surface Reconstruction do not achieve satisfactory results. Due to the great success of deep learning in many areas of computer vision, neural networks seem to be a good path to follow.

It has been shown that modeling volumetric objects in an adversarial manner results in objects that are both novel and realistic: the same concept can be applied to scene completion, generating the missing parts.

we built a generative architecture combining features of different models. On top of a standard GAN architecture, we introduced a fully convolutional variational auto encoder that is able to learn a latent representation of a 3d volumetric grid with 38 classes per voxel and that gets smoothly pushed towards a 0-centered multivariate gaussian thanks to the introduction of an adversarial loss and an encoding discriminator. We perform a sampling strategy on the scenes so that at training time we process small blocks but at evaluation time we are able to tesselate the input and test on arbitrary size at a higher resolution than state-of-the-art methods ( 3cm3).The completion achieved with the combination of the modern generative adversarial technology in a volumetric convolutional fashion and the power of variational auto encoders looks natural and highly varied.

# Data

SUNCG dataset consists of a large scale database of 3D scenes. Every scene is semantically annotated at object level. We extracted from the SUNCG dataset 773 different rooms for training and 50 for testing, represented in 3D volumetric grids. The volumetric grid is a mesh of voxels (extracted at a resolution of 3cm3) with 38 entries per voxel (each per different class of object). As input, our method requires both partial 3D scan, encoded in a volumetric grid with truncated signed distance field values, and the 3D volumetric ground truth scene, encoded in a one hot fashion. In the latter case, every voxel has 38 entries representing the classes of the objects, where only one of those is set to 1 (if the voxels contains that class of object) while the others are set to 0.
<img src="./img/SUNCGdataset.png" width="600">
<img src="./img/dcgtexample.png" width="600">
# Model
The figure below shows the architecture of our model, a 3D Variational Adversarial Auto Encoder Generative Adversarial Network.

<object data=".img/modelArchFINal.pdf" type="application/pdf" width="700px" height="700px">
    <embed src=".img/modelArchFINal.pdf">
        <p>This browser does not support PDFs.
    </embed>
</object>

<object data=".img/modelLegend.pdf" type="application/pdf" width="700px" height="700px">
    <embed src=".img/modelLegend.pdf">
        <p>This browser does not support PDFs.
    </embed>
</object>

# Results

<img src="./img/RESULT2.png" width="600">

<img src="./img/RESULT3.png" width="600">


