# <p align=center>`awesome image synthesis papers`</p>
Collection of papers in image synthesis.

# Unconditional/(Class Conditional) Image Generation
## Architecture
**Generative adversarial nets.** <br>
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*<br>
NeurIPS 2014. [[PDF](https://arxiv.org/abs/1406.2661)] [[Tutorial](https://arxiv.org/abs/1701.00160)]

`DCGAN` **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** <br>
Alec Radford, Luke Metz, Soumith Chintala. <br>
ICLR 2016. [[PDF](https://arxiv.org/abs/1511.06434)]

`PG-GAN` **Progressive Growing of GANs for Improved Quality, Stability, and Variation.** <br>
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1710.10196)]

`StyleGAN` **A Style-Based Generator Architecture for Generative Adversarial Networks.** <br>
Tero Karras, Samuli Laine, Timo Aila. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1812.04948)]

`BigGAN` **Large Scale GAN Training for High Fidelity Natural Image Synthesis.** <br>
Andrew Brock, Jeff Donahue, Karen Simonyan. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1809.11096)]

`StyleGAN2` **Analyzing and Improving the Image Quality of StyleGAN.**<br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.04958)]

`VQGAN` **Taming Transformers for High-Resolution Image Synthesis**<br>
Patrick Esser, Robin Rombach, Björn Ommer.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.09841)] [[Project](https://compvis.github.io/taming-transformers/)]

`TransGAN` **TransGAN: Two Transformers Can Make One Strong GAN, and That Can Scale Up**<br>
Yifan Jiang, Shiyu Chang, Zhangyang Wang.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2102.07074)] [[Pytorch](https://github.com/asarigun/TransGAN)]

`StyleGAN3` **Alias-Free Generative Adversarial Networks.**<br>
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2022. [[PDF](https://arxiv.org/abs/2106.12423)] [[Project](https://nvlabs.github.io/stylegan3/)]

## Objective
<!-- https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c -->
`LSGAN` **Least Squares Generative Adversarial Networks.**<br>
Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley. <br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1611.04076)]

`GGAN` **Geometric GAN** <br>
Jae Hyun Lim, Jong Chul Ye. <br>
Axiv 2017. [[PDF](https://arxiv.org/abs/1705.02894)]

`WGAN` **Wasserstein GAN**<br>
Martin Arjovsky, Soumith Chintala, Léon Bottou.<br>
ICML 2017. [[PDF](https://arxiv.org/abs/1701.07875)]

## Regularization / Limited Data

**Spectral Normalization for Generative Adversarial Networks.**<br>
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1802.05957)]

`WGAN-GP` **Improved training of wasserstein gans**<br>
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1704.00028)]

`CR-GAN` **Consistency regularization for generative adversarial networks.**<br>
Han Zhang, Zizhao Zhang, Augustus Odena, Honglak Lee. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1910.12027)]
<details>
<summary>Summary</summary>
Motivation: GANs training is unstable. Traditional regularization methods introduce non-trivial computational overheads. Discrminator is easy to focus on local features instead of semantic information. Images of different semantic objects may be close in the discriminator's feature space due to their similarity in viewpoint.<br>
Method: Restrict the discrminator's intermediate features to be consistent under data augmentations of the same image. The generator doesn't need to change.<br>
Experiment: 
(1) Augmentation details: randomly shifting the image by a few pixels and randomly flipping the image horizontally.
(2) Effect of CR: Improve FID of generated images.
(3) Ablation Study: Training with data augmentation will prevent discriminator from overfitting on training data, but not improve FID. 
The author claim this is due to consistency regularization further enforce the discriminator to learn a semantic representation.
</details>

**Differentiable Augmentation for Data-Efficient GAN Training.** <br>
Zhao Shengyu, Liu Zhijian, Lin Ji, Zhu Jun-Yan, Han Song.<br>
NeurIPS 2020.  [[PDF](https://arxiv.org/abs/2006.10738)] [[Project](https://hanlab.mit.edu/projects/data-efficient-gans/)]<br>

`ICR-GAN` **Improved consistency regularization for GANs.** <br>
Zhengli Zhao, Sameer Singh, Honglak Lee, Zizhao Zhang, Augustus Odena, Han Zhang. <br>
AAAI 2021. [[PDF](https://arxiv.org/abs/2002.04724)]
<details>
<summary>Summary</summary>
Motivation: The consistency regularization will introduce artifacts into GANs sample correponding to <br>
Method: 1. (bCR) In addition to CR,  bCR also encourage discriminator output the same feature for generated image and its augmentation. 
2. (zCR) zCR encourage discriminator insensitive to generated images with perturbed latent code, while encourage generator sensitive to that. <br>
Experiment: the augmentation to image is same as CR-GAN, the augmentation to latent vector is guassian noise.
</details>

`StyleGAN-ADA` **Training Generative Adversarial Networks with Limited Data.**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2006.06676)] [[Tensorflow](https://github.com/NVlabs/stylegan2-ada)] [[Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)]
<!-- <details>
<summary>summary</summary>
Motivation: <br>
1. 如何防止augmentation leak.
2. 数据少的时候, discriminator会过拟合, 表现在很容易区分出real和fake, 而FID很早就收敛,然后开始变差
</details> -->

**Gradient Normalization for Generative Adversarial Networks.**<br>
Yi-Lun Wu, Hong-Han Shuai, Zhi-Rui Tam, Hong-Yu Chiu. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.02235)]

**Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data.**<br>
Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2111.06849)]

## Metric
`Inception-Score/IS` **Improved Techniques for Training GANs**
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. <br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.03498)]

`FID, TTUR` **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium**<br>
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1706.08500)]

`SWD` **Sliced Wasserstein Generative Models**
Jiqing Wu, Zhiwu Huang, Dinesh Acharya, Wen Li, Janine Thoma, Danda Pani Paudel, Luc Van Gool. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1706.02631)]

## Fast Convergence
`FastGAN` **Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis**<br>
Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal. <br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.04775)]

`ProjectedGAN` **Projected GANs Converge Faster**<br>
Axel Sauer, Kashyap Chitta, Jens Müller, Andreas Geiger<br>
[[PDF](https://proceedings.neurips.cc//paper/2021/hash/9219adc5c42107c4911e249155320648-Abstract.html)] [[Project](https://sites.google.com/view/projected-gan/)] [[Pytorch](https://github.com/autonomousvision/projected_gan)]


# Image Manipulation with Deep Generative Model
## GAN Inversion

`iGAN` **Generative Visual Manipulation on the Natural Image Manifold**<br>
Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros. <br>
ECCV 2016. [[PDF](https://arxiv.org/abs/1609.03552)] [[github](https://github.com/junyanz/iGAN)]

**Neural photo editing with introspective adversarial networks**<br>
Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1609.07093)]

**Inverting The Generator of A Generative Adversarial Network.**<br>
Antonia Creswell, Anil Anthony Bharath. <br>
NeurIPS 2016 Workshop. [[PDF](https://arxiv.org/abs/1611.05644)]

`GAN Paint` **Semantic Photo Manipulation with a Generative Image Prior**<br>
David Bau, Hendrik Strobelt, William Peebles, Jonas Wulff, Bolei Zhou, Jun-Yan Zhu, Antonio Torralba.<br>
SIGGRAPH 2019. [[PDF](https://arxiv.org/abs/2005.07727)]

`GANSeeing` **Seeing What a GAN Cannot Generate.**<br>
David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, Antonio Torralba.<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1910.11626)]
<details>
<summary>summary</summary>
Summary: To see what a GAN cannot generate (mode collapse problem), this paper first inspects the distribution of semantic classes of generated images compared with groundtruth images. Sencond, by inverting images, the failure cases of image instances can be directly observed.<br>
Class Distribution-Level Mode Collapse: StyleGAN outperforms WGAN-GP.<br>
Instance Level Mode Collapse with GAN Inversion: (1) Use intermediate features instead of initial latent code as the optimization target. (2) Propose layer-wise inversion to learn the encoder for inversion, note this inversion output z coe. (3) Use restirction on z code to regularilize the inversion of intermediate feature. <br>
Experiment: (1) Directly optimization on z not work. (2) encoder + optimization works better (3) Layer-wise inversion obviously better.  <br>
Limitation: Layer-wise inversion is not performed on StyleGAN.

</details>

**Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?** <br>
Rameen Abdal, Yipeng Qin, Peter Wonka. <br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1904.03189)]

**Image2StyleGAN++: How to Edit the Embedded Images?**<br>
Rameen Abdal, Yipeng Qin, Peter Wonka. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1911.11544)]

<!-- mGANPrior -->

`IDInvert` **In-Domain GAN Inversion for Real Image Editing**<br>
Jiapeng Zhu, Yujun Shen, Deli Zhao, Bolei Zhou. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.00049)]

**Editing in Style: Uncovering the Local Semantics of GANs**<br>
Edo Collins, Raja Bala, Bob Price, Sabine Süsstrunk. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2004.14367)]

**Improving Inversion and Generation Diversity in StyleGAN using a Gaussianized Latent Space**<br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2009.06529)]

`pSp,pixel2style2pixel` **Encoding in style: a stylegan encoder for image-to-image translation.**<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2008.00951)] [[Pytorch](https://github.com/eladrich/pixel2style2pixel)]

`e4e, encode for editing` **Designing an encoder for StyleGAN image manipulation.**<br>
Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, Daniel Cohen-Or.<br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2102.02766)]

`ReStyle` **Restyle: A residual-based stylegan encoder via iterative refinement.**<br>
Yuval Alaluf, Or Patashnik, Daniel Cohen-Or. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.02699)] [[Project](https://yuval-alaluf.github.io/restyle-encoder/)]

`StyleSpace` **StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation**<br>
Zongze Wu, Dani Lischinski, Eli Shechtman. <br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.12799)]

**Collaborative Learning for Faster StyleGAN Embedding.** <br>
Shanyan Guan, Ying Tai, Bingbing Ni, Feida Zhu, Feiyue Huang, Xiaokang Yang. <br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2007.01758)]
<details>
<summary>Summary</summary>
1. Motivation: Traditional methods either use optimization based of learning based methods to get the embeded latent code. However, the optimization based method suffers from large time cost and is sensitive to initiialization. The learning based method get relative worse image quality due to the lack of direct supervision on latent code. <br>
2. This paper introduce a collaborartive training process consisting of an learnable embedding network and an optimization-based iterator to train the embedding network. For each training batch, the embedding network firstly encode the images as initialization code of the iterator, then the iterator update 100 times to optimize MSE and LPIPS loss of generated images with target image, after that the updated embedding code is used as target signal to train the embedding network with latent code distance, image-level and feature-level loss.<br>
3. The embedding network consists of a pretrained Arcface model as identity encoder, an attribute encoder built with ResBlock, the output identity feature and attribute feature are combined via linear modulation(denomarlization in SPADE). After that a Treeconnect(a sparse alterative to fully-connected layer) is used to output the final embeded code.
</details>

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing.**<br>
Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, Amit H. Bermano. <br>
(CVPR 2022?) [[PDF](https://arxiv.org/abs/2111.15666)] [[Project](https://yuval-alaluf.github.io/hyperstyle/)]

## Supervised GAN Manipulation
**GAN Dissection: Visualizing and Understanding Generative Adversarial Networks**<br>
David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B. Tenenbaum, William T. Freeman, Antonio Torralba. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1811.10597)] [[Project](http://gandissect.csail.mit.edu/)].

`InterFaceGAN` **Interpreting the Latent Space of GANs for Semantic Face Editing**<br>
Yujun Shen, Jinjin Gu, Xiaoou Tang, Bolei Zhou.
CVPR 2020. [[PDF](https://arxiv.org/abs/1907.10786)] [[Project](https://genforce.github.io/interfacegan/)]

<!-- **Only a matter of style: Age transformation using a style-based regression model.** -->
<!-- StyleFlow -->
<!-- EditGAN -->

## Unsupervised GAN Manipulation
**Unsupervised Discovery of Interpretable Directions in the GAN Latent Space.**<br>
Andrey Voynov, Artem Babenko.<br>
ICML 2020. [[PDF](https://arxiv.org/abs/2002.03754)]

`GANSpace`**GANSpace: Discovering Interpretable GAN Controls**<br>
Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, Sylvain Paris.<br>
NeurIPS 2020 [[PDF](https://arxiv.org/abs/2004.02546)] [[Pytorch](https://github.com/harskish/ganspace)]

`SeFa` **Closed-Form Factorization of Latent Semantics in GANs**<br>
Yujun Shen, Bolei Zhou. <br>
CVPR 2021 [[PDF](https://arxiv.org/abs/2007.06600)] [[Project](https://genforce.github.io/sefa/)]

**The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement**<br>
William Peebles, John Peebles, Jun-Yan Zhu, Alexei Efros, Antonio Torralba<br>
ECCV 2020 [[PDF](https://arxiv.org/abs/2008.10599)] [[Project](https://www.wpeebles.com/hessian-penalty)]


<!-- EigenGAN -->

## CLIP based
**StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**<br>
Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, Dani Lischinski<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.17249)] [[Pytorch](https://github.com/orpatashnik/StyleCLIP)]

**StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators**<br>
Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, Daniel Cohen-Or.<br>
[[PDF](https://arxiv.org/abs/2108.00946)] [[Project](https://stylegan-nada.github.io/)]

**CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders.**<br>
Kevin Frans, L.B. Soros, Olaf Witkowski.<br>
Arxiv 2021. [[PDF](https://arxiv.org/abs/2106.14843)]


# Image-to-Image Translation

## Style Transfer

## Others
`GANgealing`**GAN-Supervised Dense Visual Alignment**<br>
William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, Eli Shechtman.<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.05143)]

# Text-to-Image Synthesis

# Others

## Single Image Generation

`DIP` **Deep Image Prior**<br>
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. <br>
CVPR 2018 [[PDF](https://arxiv.org/abs/1711.10925)] [[Project](https://dmitryulyanov.github.io/deep_image_prior)]

`SinGAN` **SinGAN: Learning a Generative Model from a Single Natural Image**<br>
Tamar Rott Shaham, Tali Dekel, Tomer Michaeli. <br>
ICCV 2019 Best Paper. [[PDF](https://arxiv.org/abs/1905.01164)] [[Project](https://tamarott.github.io/SinGAN.htm)]


`TuiGAN` **TuiGAN: Learning Versatile Image-to-Image Translation with Two Unpaired Images** <br>
Jianxin Lin, Yingxue Pang, Yingce Xia, Zhibo Chen, Jiebo Luo. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.04634)]


`DeepSIM` **Image Shape Manipulation from a Single Augmented Training Sample**
Yael Vinker, Eliahu Horwitz, Nir Zabari , Yedid Hoshen. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2007.01289)] [[Project](https://www.vision.huji.ac.il/deepsim/)] [[Pytorch](https://github.com/eliahuhorwitz/DeepSIM)]