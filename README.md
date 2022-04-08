# <p align=center>`awesome image synthesis papers`</p>
Collection of papers in image synthesis.

# Unconditional/(Class Conditional) Image Generation
## GAN Architecture
```mermaid
flowchart TB
  GAN[VanillaGAN, 2014] == architecture tricks --> DCGAN[DCGAN, 2016]
  DCGAN == Progressive growing --> PG[PG-GAN, 2018]
  PG --> BigGAN[BigGAN, 2019]
  PG == AdaIN, mapping network --> SG1[StyleGAN, 2019]
  SG1 == Weight demodulation --> SG2[StyleGAN2, 2020]
  SG2 == Translate and rotate equivariance --> SG3[StyleGAN3, 2021]
  DCGAN == Autoregressive transformer \n for vison tokens --> VQGAN
  VQGAN == transformers architecture \n of  generator and discriminator --> TransGAN
```
**Generative adversarial nets.** <br>
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*<br>
NeurIPS 2014. [[PDF](https://arxiv.org/abs/1406.2661)] [[Tutorial](https://arxiv.org/abs/1701.00160)]

`DCGAN` **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** <br>
Alec Radford, Luke Metz, Soumith Chintala. <br>
ICLR 2016. [[PDF](https://arxiv.org/abs/1511.06434)] Cited:`9219`

`PG-GAN` **Progressive Growing of GANs for Improved Quality, Stability, and Variation.** <br>
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1710.10196)] Cited:`3726`

`StyleGAN` **A Style-Based Generator Architecture for Generative Adversarial Networks.** <br>
Tero Karras, Samuli Laine, Timo Aila. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1812.04948)] Cited:`3269`

`BigGAN` **Large Scale GAN Training for High Fidelity Natural Image Synthesis.** <br>
Andrew Brock, Jeff Donahue, Karen Simonyan. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1809.11096)] Cited:`2543`

`StyleGAN2` **Analyzing and Improving the Image Quality of StyleGAN.**<br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.04958)] Cited:`1421`

`VQGAN` **Taming Transformers for High-Resolution Image Synthesis**<br>
Patrick Esser, Robin Rombach, Björn Ommer.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.09841)] [[Project](https://compvis.github.io/taming-transformers/)] Cited:`177`

`TransGAN` **TransGAN: Two Transformers Can Make One Strong GAN, and That Can Scale Up**<br>
Yifan Jiang, Shiyu Chang, Zhangyang Wang.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2102.07074)] [[Pytorch](https://github.com/asarigun/TransGAN)] Cited:`27`

`StyleGAN3` **Alias-Free Generative Adversarial Networks.**<br>
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2106.12423)] [[Project](https://nvlabs.github.io/stylegan3/)] Cited:`86`

**StyleSwin: Transformer-based GAN for High-resolution Image Generation**<br>
Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, Baining Guo<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.10762)] Cited:`4`

## GAN Objective
**A Large-Scale Study on Regularization and Normalization in GANs**<br>
*Karol Kurach, Mario Lucic, Xiaohua Zhai, Marcin Michalski, Sylvain Gelly*<br>
ICML 2019. [[PDF](https://arxiv.org/abs/1807.04720)] Cited:`128`

`EB-GAN` **Energy-based Generative Adversarial Networks**<br>
Junbo Zhao, Michael Mathieu, Yann LeCun.<br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1609.03126)] Cited:`859`

**Towards Principled Methods for Training Generative Adversarial Networks**<br>
Martin Arjovsky, Léon Bottou<br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1701.04862)] Cited:`1375`

<!-- https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c -->
`LSGAN` **Least Squares Generative Adversarial Networks.**<br>
Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley. <br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1611.04076)] Cited:`2674`

`WGAN` **Wasserstein GAN**<br>
Martin Arjovsky, Soumith Chintala, Léon Bottou.<br>
ICML 2017. [[PDF](https://arxiv.org/abs/1701.07875)] Cited:`2728`

`GGAN` **Geometric GAN** <br>
Jae Hyun Lim, Jong Chul Ye. <br>
Axiv 2017. [[PDF](https://arxiv.org/abs/1705.02894)] Cited:`176`

`AC-GAN` **Conditional Image Synthesis With Auxiliary Classifier GANs** <br>
Augustus Odena, Christopher Olah, Jonathon Shlens. <br>
ICML 2017. [[PDF](https://arxiv.org/abs/1610.09585)] Cited:`2030`

**cGANs with Projection Discriminator**<br>
Takeru Miyato, Masanori Koyama. <br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1802.05637)] Cited:`526`

`S³-GAN` **High-Fidelity Image Generation With Fewer Labels**<br>
Mario Lucic*, Michael Tschannen*, Marvin Ritter*, Xiaohua Zhai, Olivier Bachem, Sylvain Gelly. <br>
ICML 2019. [[PDF](https://arxiv.org/abs/1903.02271)] [[Tensorflow](https://github.com/google/compare_gan)] Cited:`97`

## Autoencoder-based framework
`VAE` **Auto-Encoding Variational Bayes.**<br>
Diederik P.Kingma, Max Welling.<br>
ICLR 2014. [[PDF](https://arxiv.org/abs/1312.6114)] Cited:`14936`

`AAE` **Adversarial Autoencoders.**<br>
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey.<br>
arxiv 2015. [[PDF](https://arxiv.org/abs/1511.05644)] Cited:`1329`

`VAE/GAN` **Autoencoding beyond pixels using a learned similarity metric.** <br>
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther.<br>
ICML 2016. [[PDF](https://arxiv.org/abs/1512.09300)] Cited:`1377`

`VampPrior` **VAE with a VampPrior** <br>
Jakub M. Tomczak, Max Welling.<br>
AISTATS 2018. [[PDF](https://arxiv.org/abs/1705.07120)] [[Pytorch](https://github.com/jmtomczak/vae_vampprior)] Cited:`367`

`BiGAN` **Adversarial Feature Learning**<br>
Jeff Donahue, Philipp Krähenbühl, Trevor Darrell. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1605.09782)] Cited:`1307`

`AIL` **Adversarial Learned Inference**<br>
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1606.00704)] Cited:`1044`

`VEEGAN` **Veegan: Reducing mode collapse in gans using implicit variational learning.**<br>
Akash Srivastava, Lazar Valkov, Chris Russell, Michael U. Gutmann, Charles Sutton.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1705.07761)] [[Github](https://github.com/akashgit/VEEGAN)] Cited:`392`

`AGE` **Adversarial Generator-Encoder Networks.**<br>
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky.<br>
AAAI 2018. [[PDF](https://arxiv.org/abs/1704.02304)] [[Pytorch](https://github.com/DmitryUlyanov/AGE)] Cited:`106`

`IntroVAE` **IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis.**<br>
Huaibo Huang, Zhihang Li, Ran He, Zhenan Sun, Tieniu Tan. <br>
NeurIPS 2018. [[PDF](https://arxiv.org/abs/1807.06358)] Cited:`133`

`ALAE` **Adversarial Latent Autoencoders**<br>
Stanislav Pidhorskyi, Donald Adjeroh, Gianfranco Doretto. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2004.04467)] Cited:`116`

## Disentangled Image Generation
`DC-IGN` **Deep Convolutional Inverse Graphics Network**<br>
Tejas D. Kulkarni, Will Whitney, Pushmeet Kohli, Joshua B. Tenenbaum. <br>
NeurIPS 2015. [[PDF](Deep Convolutional Inverse Graphics Network)]

`InfoGAN` **InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets**<br>
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel. <br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.03657)] Cited:`2961`

`Beta-VAE` **beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**<br>
I. Higgins, L. Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, M. Botvinick, S. Mohamed, Alexander Lerchner. <br>
ICLR 2017. [[PDF](https://openreview.net/forum?id=Sy2fzU9gl)]

`AnnealedVAE` **Understanding disentangling in β-VAE**<br>
Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, Alexander Lerchner. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1804.03599)] Cited:`261`

`Factor-VAE` **Disentangling by Factorising**<br>
Hyunjik Kim, Andriy Mnih. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1802.05983)] Cited:`732`

`DCI` **A framework for the quantitative evaluation of disentangled representations.**<br>
*Cian Eastwood, Christopher K. I. Williams*
ICLR 2018. [[PDF](https://openreview.net/pdf?id=By-7dz-AZ)] 

**Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.**<br>
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem.<br>
ICML(best paper award) 2019. [[PDF](https://arxiv.org/abs/1811.12359)] Cited:`676`

## Regularization / Limited Data
`WGAN-GP` **Improved training of wasserstein gans**<br>
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1704.00028)] Cited:`5419`

**The Numerics of GANs**<br>
*Lars Mescheder, Sebastian Nowozin, Andreas Geiger*<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1705.10461)] Cited:`331`

`R1-regularization` **Which Training Methods for GANs do actually Converge?**<br>
*Lars Mescheder, Andreas Geiger, Sebastian Nowozin.*<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1801.04406)] Cited:`769`

`SN-GAN` **Spectral Normalization for Generative Adversarial Networks.**<br>
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1802.05957)] Cited:`2610`

`CR-GAN` **Consistency regularization for generative adversarial networks.**<br>
Han Zhang, Zizhao Zhang, Augustus Odena, Honglak Lee. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1910.12027)] Cited:`126`
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
NeurIPS 2020.  [[PDF](https://arxiv.org/abs/2006.10738)] [[Project](https://hanlab.mit.edu/projects/data-efficient-gans/)]<br> Cited:`171`

`ICR-GAN` **Improved consistency regularization for GANs.** <br>
Zhengli Zhao, Sameer Singh, Honglak Lee, Zizhao Zhang, Augustus Odena, Han Zhang. <br>
AAAI 2021. [[PDF](https://arxiv.org/abs/2002.04724)] Cited:`53`
<details>
<summary>Summary</summary>
Motivation: The consistency regularization will introduce artifacts into GANs sample correponding to <br>
Method: 1. (bCR) In addition to CR,  bCR also encourage discriminator output the same feature for generated image and its augmentation. 
2. (zCR) zCR encourage discriminator insensitive to generated images with perturbed latent code, while encourage generator sensitive to that. <br>
Experiment: the augmentation to image is same as CR-GAN, the augmentation to latent vector is guassian noise.
</details>

`StyleGAN-ADA` **Training Generative Adversarial Networks with Limited Data.**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2006.06676)] [[Tensorflow](https://github.com/NVlabs/stylegan2-ada)] [[Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)] Cited:`387`
<!-- <details>
<summary>summary</summary>
Motivation: <br>
1. 如何防止augmentation leak.
2. 数据少的时候, discriminator会过拟合, 表现在很容易区分出real和fake, 而FID很早就收敛,然后开始变差
</details> -->

**Gradient Normalization for Generative Adversarial Networks.**<br>
Yi-Lun Wu, Hong-Han Shuai, Zhi-Rui Tam, Hong-Yu Chiu. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.02235)] Cited:`4`

**Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data.**<br>
Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2111.06849)] Cited:`1`

## Metric
`Inception-Score/IS` **Improved Techniques for Training GANs**
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. <br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.03498)] Cited:`5319`

`FID, TTUR` **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium**<br>
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1706.08500)] Cited:`4106`

`SWD` **Sliced Wasserstein Generative Models**
Jiqing Wu, Zhiwu Huang, Dinesh Acharya, Wen Li, Janine Thoma, Danda Pani Paudel, Luc Van Gool. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1706.02631)] Cited:`0`

## Fast Convergence
`FastGAN` **Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis**<br>
Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal. <br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.04775)] Cited:`31`

`ProjectedGAN` **Projected GANs Converge Faster**<br>
Axel Sauer, Kashyap Chitta, Jens Müller, Andreas Geiger<br>
[[PDF](https://arxiv.org/abs/2111.01007)] [[Project](https://sites.google.com/view/projected-gan/)] [[Pytorch](https://github.com/autonomousvision/projected_gan)] Cited:`8`

## GAN Adaptation
**Transferring GANs: generating images from limited data.**<br>
Yaxing Wang, Chenshen Wu, Luis Herranz, Joost van de Weijer, Abel Gonzalez-Garcia, Bogdan Raducanu.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1805.01677)] Cited:`116`

**Image Generation From Small Datasets via Batch Statistics Adaptation.**<br>
Atsuhiro Noguchi, Tatsuya Harada.<br>
ICCV 2019 [[PDF](https://arxiv.org/abs/1904.01774)] Cited:`67`

**Freeze Discriminator: A Simple Baseline for Fine-tuning GANs.**<br>
Sangwoo Mo, Minsu Cho, Jinwoo Shin.<br>
CVPRW 2020 [[PDF](https://arxiv.org/abs/2002.10964)] [[Pytorch](https://github.com/sangwoomo/FreezeD)] Cited:`52`

**Resolution dependant GAN interpolation for controllable image synthesis between domains.**<br>
Justin N. M. Pinkney, Doron Adler<br>
NeruIPS workshop 2020. [[PDF](https://arxiv.org/abs/2010.05334)] Cited:`30`

**Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs**<br>
Sangwoo Mo, Minsu Cho, Jinwoo Shin<br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2002.10964)] Cited:`52`

**Unsupervised image-to-image translation via pre-trained StyleGAN2 network**<br>
_Jialu Huang, Jing Liao, Sam Kwong_<br>
TMM 2021. [[PDF](https://arxiv.org/abs/2010.05713)] Cited:`11`

**Few-shot Adaptation of Generative Adversarial Networks**<br>
Esther Robb, Wen-Sheng Chu, Abhishek Kumar, Jia-Bin Huang.<br>
arxiv 2020 [[PDF](https://arxiv.org/abs/2010.11943)] Cited:`21`

**AgileGAN: stylizing portraits by inversion-consistent transfer learning.**<br>
_Guoxian Song, Linjie Luo, Jing Liu, Wan-Chun Ma, Chunpong Lai, Chuanxia Zheng, Tat-Jen Cham_<br>
TOG/SIGGRAPH 2021. [[PDF](https://guoxiansong.github.io/homepage/paper/AgileGAN.pdf)] [[Project](https://guoxiansong.github.io/homepage/agilegan.html)]

**Few-shot Image Generation via Cross-domain Correspondence**<br>
Utkarsh Ojha, Yijun Li, Jingwan Lu, Alexei A. Efros, Yong Jae Lee, Eli Shechtman, Richard Zhang.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.06820)] Cited:`28`

**StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators**<br>
Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, Daniel Cohen-Or.<br>
arxiv 2021 [[PDF](https://arxiv.org/abs/2108.00946)] [[Project](https://stylegan-nada.github.io/)] Cited:`33`

**Stylealign: Analysis and Applications of Aligned StyleGAN Models**<br>
_Zongze Wu, Yotam Nitzan, Eli Shechtman, Dani Lischinski_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2110.11323)] Cited:`8`

**Mind the Gap: Domain Gap Control for Single Shot Domain Adaptation for Generative Adversarial Networks**<br>
_Peihao Zhu, Rameen Abdal, John Femiani, Peter Wonka_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2110.08398)] Cited:`10`

**Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment**<br>
_Jiayu Xiao, Liang Li, Chaofei Wang, Zheng-Jun Zha, Qingming Huang_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.04121)] Cited:`0`

**JoJoGAN: One Shot Face Stylization**<br>
_Min Jin Chong, David Forsyth_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2112.11641)] Cited:`4`

**When why and which pretrained GANs are useful?**<br>
*Timofey Grigoryev, Andrey Voynov, Artem Babenko*<br>
ICLR 2022. [[PDF](https://openreview.net/forum?id=4Ycr8oeCoIh)]

**CtlGAN: Few-shot Artistic Portraits Generation with Contrastive Transfer Learning**<br>
_Yue Wang, Ran Yi, Ying Tai, Chengjie Wang, and Lizhuang Ma_

## Other Generative Models
**Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space**<br>
_Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, Jason Yosinski_<br>
CVPR 2017. [[PDF](https://arxiv.org/abs/1612.00005)] Cited:`508`

**Optimizing the Latent Space of Generative Networks**<br>
Piotr Bojanowski, Armand Joulin, David Lopez-Paz, Arthur Szlam<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1707.05776)] Cited:`256`

## Latent Interpolation
**Sampling generative networks: Notes on a few effective techniques.**<br>
Tom White.<br>
arxiv 2016 [[PDF](https://arxiv.org/abs/1609.04468v2)] Cited:`112`

**Latent space oddity: on the curvature of deep generative models**<br>
Georgios Arvanitidis, Lars Kai Hansen, Søren Hauberg.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1710.11379)] Cited:`136`

**Feature-Based Metrics for Exploring the Latent Space of Generative Models**<br>
Samuli Laine.<br>
ICLR 2018 Workshop. [[PDF](https://openreview.net/forum?id=BJslDBkwG)]

# Image Manipulation with Deep Generative Model
## GAN Inversion

`iGAN` **Generative Visual Manipulation on the Natural Image Manifold**<br>
Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros. <br>
ECCV 2016. [[PDF](https://arxiv.org/abs/1609.03552)] [[github](https://github.com/junyanz/iGAN)] Cited:`985`

`IcGAN` **Invertible Conditional GANs for image editing**<br>
Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, Jose M. Álvarez<br>
NIPS 2016 Workshop. [[PDF](https://arxiv.org/abs/1611.06355)] Cited:`444`

**Neural photo editing with introspective adversarial networks**<br>
Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1609.07093)] Cited:`339`

**Inverting The Generator of A Generative Adversarial Network.**<br>
Antonia Creswell, Anil Anthony Bharath. <br>
NeurIPS 2016 Workshop. [[PDF](https://arxiv.org/abs/1611.05644)] Cited:`174`

`GAN Paint` **Semantic Photo Manipulation with a Generative Image Prior**<br>
David Bau, Hendrik Strobelt, William Peebles, Jonas Wulff, Bolei Zhou, Jun-Yan Zhu, Antonio Torralba.<br>
SIGGRAPH 2019. [[PDF](https://arxiv.org/abs/2005.07727)] Cited:`191`

`GANSeeing` **Seeing What a GAN Cannot Generate.**<br>
David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, Antonio Torralba.<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1910.11626)] Cited:`133`
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
ICCV 2019. [[PDF](https://arxiv.org/abs/1904.03189)] Cited:`367`

**Image2StyleGAN++: How to Edit the Embedded Images?**<br>
Rameen Abdal, Yipeng Qin, Peter Wonka. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1911.11544)] Cited:`163`

<!-- mGANPrior -->

`IDInvert` **In-Domain GAN Inversion for Real Image Editing**<br>
Jiapeng Zhu, Yujun Shen, Deli Zhao, Bolei Zhou. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.00049)] Cited:`196`
<details>
<summary>summary</summary>
Motivation: Traditional GAN Inversion method train the encoder in the latent space via optimizing the distance to  |E(G(z))-z|. However, the gradient to encoder is agnostic about the semantic distribution of generator's latent space. (For example, latent code far from mean vector is less editable.) 
This paper first train a domain-guided encoder, and then propose domain-regularized optimization by involving the encoder as a regularizer to finetune the code produced by the encoder and better recover the target image.<br>
Method:
(1) Objective for training encoder: MSE loss and perceptual loss for reconstructed real image, adversarial loss.
(2) Objective for refining embeded code: perceptual loss and MSE for reconstructed image, distance from to inverted code by encoder as regularization. <br>

Experiment:
(1) semantic analysis of inverted code: Train attribute boundry of inverted code with InterFaceGAN, compared with Image2StyleGAN, the Precision-Recall Curve performs betters.
(2) Inversion Quality: Compared by FID, SWD, MSE and visual quality.
(3) Application: Image Interpolation, Semantic Manipulation, Semantic Diffusion(Inversion of Composed image and then optimize with only front image), Style Mixing
(4) Ablation Study: Larger weight for encoder bias the optimization towards the domain constraint such that the inverted codes are more semantically meaningful. Instead, the cost is that the target image cannot be ideally recovered for per-pixel values.
</details>

**Editing in Style: Uncovering the Local Semantics of GANs**<br>
Edo Collins, Raja Bala, Bob Price, Sabine Süsstrunk. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2004.14367)] [[Pytorch](https://github.com/cyrilzakka/GANLocalEditing)] Cited:`96`
<details>
<summary>summary</summary>
StyleGAN's style code controls the global style of images, so how to make local manipulation based on style code? 
Remeber that the style code is to modulate the variance of intermediate variations, different channels control different local semantic elements like noise and eyes.
So we can identity the channel most correlated to the region of interest for local manipulation, and then replace value of source image style code of that channel with corresponding target channel.<br>
Details: The corresponding between RoI and channel is measured by feature map magnitude within each cluster, and the cluster is calculated from spherical k-means on features in 32x32 layer.
Limitation: This paper actually does local semantic swap, and interpolation is not available.<br>
</details>

**Improving Inversion and Generation Diversity in StyleGAN using a Gaussianized Latent Space**<br>
_Jonas Wulff, Antonio Torralba_<br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2009.06529)] Cited:`20`

**Improved StyleGAN Embedding: Where are the Good Latents?**<br>
Peihao Zhu, Rameen Abdal, Yipeng Qin, John Femiani, Peter Wonka<br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2012.09036)] Cited:`24`

`pix2latent` **Transforming and Projecting Images into Class-conditional Generative Networks**<br>
Minyoung Huh,Richard Zhang,Jun-Yan Zhu,Sylvain Paris,Aaron Hertzmann<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2005.01703)] Cited:`41`

`pSp,pixel2style2pixel` **Encoding in style: a stylegan encoder for image-to-image translation.**<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2008.00951)] [[Pytorch](https://github.com/eladrich/pixel2style2pixel)] Cited:`194`

`e4e, encode for editing` **Designing an encoder for StyleGAN image manipulation.**<br>
Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, Daniel Cohen-Or.<br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2102.02766)] Cited:`89`

`ReStyle` **Restyle: A residual-based stylegan encoder via iterative refinement.**<br>
Yuval Alaluf, Or Patashnik, Daniel Cohen-Or. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.02699)] [[Project](https://yuval-alaluf.github.io/restyle-encoder/)] Cited:`48`


**Collaborative Learning for Faster StyleGAN Embedding.** <br>
Shanyan Guan, Ying Tai, Bingbing Ni, Feida Zhu, Feiyue Huang, Xiaokang Yang. <br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2007.01758)] Cited:`42`
<details>
<summary>Summary</summary>
1. Motivation: Traditional methods either use optimization based of learning based methods to get the embeded latent code. However, the optimization based method suffers from large time cost and is sensitive to initiialization. The learning based method get relative worse image quality due to the lack of direct supervision on latent code. <br>
2. This paper introduce a collaborartive training process consisting of an learnable embedding network and an optimization-based iterator to train the embedding network. For each training batch, the embedding network firstly encode the images as initialization code of the iterator, then the iterator update 100 times to optimize MSE and LPIPS loss of generated images with target image, after that the updated embedding code is used as target signal to train the embedding network with latent code distance, image-level and feature-level loss.<br>
3. The embedding network consists of a pretrained Arcface model as identity encoder, an attribute encoder built with ResBlock, the output identity feature and attribute feature are combined via linear modulation(denomarlization in SPADE). After that a Treeconnect(a sparse alterative to fully-connected layer) is used to output the final embeded code.
</details>

**Pivotal Tuning for Latent-based Editing of Real Images**<br>
Daniel Roich, Ron Mokady, Amit H. Bermano, Daniel Cohen-Or. <br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2106.05744)] Cited:`28`

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing.**<br>
Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, Amit H. Bermano. <br>
CVPR 2022 [[PDF](https://arxiv.org/abs/2111.15666)] [[Project](https://yuval-alaluf.github.io/hyperstyle/)] Cited:`7`

**High-Fidelity GAN Inversion for Image Attribute Editing**<br>
Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, Qifeng Chen. <br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2109.06590)] Cited:`16`

## Supervised GAN Manipulation
**GAN Dissection: Visualizing and Understanding Generative Adversarial Networks**<br>
David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B. Tenenbaum, William T. Freeman, Antonio Torralba. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1811.10597)] [[Project](http://gandissect.csail.mit.edu/)]. Cited:`239`

**On the "steerability" of generative adversarial networks.**<br>
Ali Jahanian, Lucy Chai, Phillip Isola. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1907.07171)] [[Project](https://ali-design.github.io/gan_steerability/)] [[Pytorch](https://github.com/ali-design/gan_steerability)] Cited:`199`

**Controlling generative models with continuous factors of variations.**<br>
Antoine Plumerault, Hervé Le Borgne, Céline Hudelot. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/2001.10238)] Cited:`62`

`InterFaceGAN` **Interpreting the Latent Space of GANs for Semantic Face Editing**<br>
Yujun Shen, Jinjin Gu, Xiaoou Tang, Bolei Zhou. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1907.10786)] [[Project](https://genforce.github.io/interfacegan/)] Cited:`374`

**Enjoy your editing: Controllable gans for image editing via latent space navigation**<br>
Peiye Zhuang, Oluwasanmi Koyejo, Alexander G. Schwing<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2102.01187)] Cited:`16`

**Only a matter of style: Age transformation using a style-based regression model.**<br>
Yuval Alaluf, Or Patashnik, Daniel Cohen-Or<br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2102.02754)] Cited:`20`

**Discovering Interpretable Latent Space Directions of GANs Beyond Binary Attributes.**<br>
*Huiting Yang, Liangyu Chai, Qiang Wen, Shuang Zhao, Zixun Sun, Shengfeng He.*<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Discovering_Interpretable_Latent_Space_Directions_of_GANs_Beyond_Binary_Attributes_CVPR_2021_paper.pdf)]

`StyleSpace` **StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation**<br>
Zongze Wu, Dani Lischinski, Eli Shechtman. <br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.12799)] Cited:`91`

`StyleFlow` **StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows** <br>
Rameen Abdal, Peihao Zhu, Niloy Mitra, Peter Wonka. <br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2008.02401)] Cited:`128`

**A Latent Transformer for Disentangled Face Editing in Images and Videos.**<br>
*Xu Yao, Alasdair Newson, Yann Gousseau, Pierre Hellier.*<br>
ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_A_Latent_Transformer_for_Disentangled_Face_Editing_in_Images_and_ICCV_2021_paper.html)] [[ArXiV](https://arxiv.org/abs/2106.11895)] [[Github](https://github.com/InterDigitalInc/latent-transformer)] Cited:`9`

**Controllable and Compositional Generation with Latent-Space Energy-Based Models.**<br>
*Weili Nie, Arash Vahdat, Anima Anandkumar.*<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2110.10873)] Cited:`2`

`EditGAN` **EditGAN: High-Precision Semantic Image Editing**<br>
Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, Sanja Fidler. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2111.03186)] Cited:`11`

`StyleFusion` **StyleFusion: A Generative Model for Disentangling Spatial Segments**<br>
*Omer Kafri, Or Patashnik, Yuval Alaluf, Daniel Cohen-Or*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2107.07437)] Cited:`7`


## Unsupervised GAN Manipulation
```mermaid
flowchart TD
  root(Unsupervised GAN Manipulation) --> A(Mutual inforamtion)
  root --> B[Generator Parameter]
  root --> C[Training Regularization]

  A --> E[Unsupervised Discovery. Voynov. ICML 2020]
  InfoGAN == on pretrained network --> E
  E == RBF Path --> Warped[WarpedGANSpace. Tzelepis. ICCV 2021]
  E == Parameter Space --> NaviGAN[NaviGAN. Cherepkov. CVPR 2021]
  E == Contrastive Loss --> DisCo[Disco. Ren. ICLR 2022]

  B == PCA on Intermediate/W space --> GANSpace[GANSpace. Härkönen. NIPS 2020.]
  GANSpace == Closed-form Factorization of Weight --> SeFa[SeFa. Shen. CVPR 2021.]
  GANSpace == Spatial Transformation \n on intermediate Feature --> GANS[GAN Steerability. Eliezer. ICLR 2021]

  SeFa == Variation for intermediate features --> VisualConcept[Visual Concept Vocabulary. Schwettmann. ICCV 2021]
```
**Unsupervised Discovery of Interpretable Directions in the GAN Latent Space.**<br>
Andrey Voynov, Artem Babenko.<br>
ICML 2020. [[PDF](https://arxiv.org/abs/2002.03754)] Cited:`131`

`GANSpace`**GANSpace: Discovering Interpretable GAN Controls**<br>
Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, Sylvain Paris.<br>
NeurIPS 2020 [[PDF](https://arxiv.org/abs/2004.02546)] [[Pytorch](https://github.com/harskish/ganspace)] Cited:`233`

**The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement**<br>
William Peebles, John Peebles, Jun-Yan Zhu, Alexei Efros, Antonio Torralba<br>
ECCV 2020 [[PDF](https://arxiv.org/abs/2008.10599)] [[Project](https://www.wpeebles.com/hessian-penalty)] Cited:`45`

**The Geometry of Deep Generative Image Models and its Applications**<br>
Binxu Wang, Carlos R. Ponce.<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.06006)] Cited:`9`

**GAN Steerability without optimization.**<br>
Nurit Spingarn-Eliezer, Ron Banner, Tomer Michaeli<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2012.05328)] Cited:`19`

**The Geometry of Deep Generative Image Models and its Applications**<br>
Binxu Wang, Carlos R. Ponce<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.06006)] Cited:`9`

`SeFa` **Closed-Form Factorization of Latent Semantics in GANs**<br>
Yujun Shen, Bolei Zhou. <br>
CVPR 2021 [[PDF](https://arxiv.org/abs/2007.06600)] [[Project](https://genforce.github.io/sefa/)] Cited:`141`

`NaviGAN` **Navigating the GAN Parameter Space for Semantic Image Editing**<br>
Anton Cherepkov, Andrey Voynov, Artem Babenko.<br>
CVPR 2021 [[PDF](https://arxiv.org/abs/2011.13786)] [[Pytorch](https://github.com/yandex-research/navigan)] Cited:`12`

**EigenGAN: Layer-Wise Eigen-Learning for GANs.**<br>
*Zhenliang He, Meina Kan, Shiguang Shan.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.12476)] [[Github](https://github.com/LynnHo/EigenGAN-Tensorflow)] Cited:`13`

**Toward a Visual Concept Vocabulary for GAN Latent Space.**<br>
*Sarah Schwettmann, Evan Hernandez, David Bau, Samuel Klein, Jacob Andreas, Antonio Torralba*.<br>
ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Schwettmann_Toward_a_Visual_Concept_Vocabulary_for_GAN_Latent_Space_ICCV_2021_paper.html)] [[Project](https://visualvocab.csail.mit.edu/)]

**WarpedGANSpace: Finding Non-linear RBF Paths in GAN Latent Space.**<br>
*Christos Tzelepis, Georgios Tzimiropoulos, Ioannis Patras.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.13357)] [[Github](https://github.com/chi0tzp/WarpedGANSpace)] Cited:`8`

**OroJaR: Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation.**<br>
*Yuxiang Wei, Yupeng Shi, Xiao Liu, Zhilong Ji, Yuan Gao, Zhongqin Wu, Wangmeng Zuo.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2108.07668)] [[Github](https://github.com/csyxwei/OroJaR)] Cited:`5`

**Optimizing Latent Space Directions For GAN-based Local Image Editing.**<br>
*Ehsan Pajouheshgar, Tong Zhang, Sabine Süsstrunk.*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2111.12583)] [[Pytorch](https://github.com/IVRL/LELSD)] Cited:`0`

**Discovering Density-Preserving Latent Space Walks in GANs for Semantic Image Transformations.**<br>
*Guanyue Li, Yi Liu, Xiwen Wei, Yang Zhang, Si Wu, Yong Xu, Hau San Wong.*<br>
ACM MM 2021. [[PDF](https://dl.acm.org/doi/abs/10.1145/3474085.3475293)]

**Disentangled Representations from Non-Disentangled Models**<br>
*Valentin Khrulkov, Leyla Mirvakhabova, Ivan Oseledets, Artem Babenko*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2102.06204)] Cited:`3`

**Do Not Escape From the Manifold: Discovering the Local Coordinates on the Latent Space of GANs.**<br>
*Jaewoong Choi, Changyeon Yoon, Junho Lee, Jung Ho Park, Geonho Hwang, Myungjoo Kang.*<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2106.06959)] Cited:`2`

`Disco` **Learning Disentangled Representation by Exploiting Pretrained Generative Models: A Contrastive Learning View**<br>
*Xuanchi Ren, Tao Yang, Yuwang Wang, Wenjun Zeng*<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2102.10543)] Cited:`1`

**Rayleigh EigenDirections (REDs): GAN latent space traversals for multidimensional features.**<br>
*Guha Balakrishnan, Raghudeep Gadde, Aleix Martinez, Pietro Perona.*<br>
arxiv 2022. [[PDF](https://arxiv.org/pdf/2201.10423.pdf)]

**Region-Based Semantic Factorization in GANs**<br>
*Jiapeng Zhu, Yujun Shen, Yinghao Xu, Deli Zhao, Qifeng Chen.*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2202.09649)] Cited:`1`

**Fantastic Style Channels and Where to Find Them: A Submodular Framework for Discovering Diverse Directions in GANs**<br>


## CLIP based
**StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**<br>
Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, Dani Lischinski<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.17249)] [[Pytorch](https://github.com/orpatashnik/StyleCLIP)] Cited:`114`

**CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders.**<br>
Kevin Frans, L.B. Soros, Olaf Witkowski.<br>
Arxiv 2021. [[PDF](https://arxiv.org/abs/2106.14843)] Cited:`24`

**CLIP2StyleGAN: Unsupervised Extraction of StyleGAN Edit Directions.**<br>
*Omer Kafri, Or Patashnik, Yuval Alaluf, and Daniel Cohen-Or*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.05219)] Cited:`3`

**FEAT: Face Editing with Attention**<br>
Xianxu Hou, Linlin Shen, Or Patashnik, Daniel Cohen-Or, Hui Huang<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2202.02713)] Cited:`0`

## Inversion-based Animation
**A good image generator is what you need for high-resolution video synthesis**<br>
Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, Sergey Tulyakov. <br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2104.15069)] Cited:`20`

**Latent Image Animator: Learning to animate image via latent space navigation.**<br>
*Yaohui Wang, Di Yang, Francois Bremond, Antitza Dantcheva.*<br>
ICLR 2022. [[PDF](https://openreview.net/forum?id=7r6kDq0mK_)]

# Image-to-Image Translation
## Supervised Image Translation

`pix2pix` **Image-to-Image Translation with Conditional Adversarial Networks**<br>
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros.<br>
CVPR 2017. [[PDF](https://arxiv.org/abs/1611.07004)]

### Semantic Image Synthesis

`CRN` **Photographic Image Synthesis with Cascaded Refinement Networks**<br>
Qifeng Chen, Vladlen Koltun. <br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1707.09405)] Cited:`701`

`pix2pixHD` **High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs**<br>
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro.<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.11585)] Cited:`2087`

`SPADE` **Semantic Image Synthesis with Spatially-Adaptive Normalization**<br>
Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu.<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.07291)] Cited:`1094`

`SEAN` **SEAN: Image Synthesis with Semantic Region-Adaptive Normalization**<br>
Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka.<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1911.12861)] Cited:`160`

**You Only Need Adversarial Supervision for Semantic Image Synthesis**<br>
Vadim Sushko, Edgar Schönfeld, Dan Zhang, Juergen Gall, Bernt Schiele, Anna Khoreva.<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2012.04781)] Cited:`27`

**Diverse Semantic Image Synthesis via Probability Distribution Modeling**<br>
Zhentao Tan, Menglei Chai, Dongdong Chen, Jing Liao, Qi Chu, Bin Liu, Gang Hua, Nenghai Yu.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2103.06878)] Cited:`5`

**Efficient Semantic Image Synthesis via Class-Adaptive Normalization**<br>
Zhentao Tan, Dongdong Chen, Qi Chu, Menglei Chai, Jing Liao, Mingming He, Lu Yuan, Gang Hua, Nenghai Yu.<br>
TPAMI 2021. [[PDF](https://arxiv.org/pdf/2012.04644.pdf)]

**Spatially-adaptive pixelwise networks for fast image translation.**<br>
Tamar Rott Shaham, Michael Gharbi, Richard Zhang, Eli Shechtman, Tomer Michaeli<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.02992)] Cited:`13`

**High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network**<br>
Jie Liang, Hui Zeng, Lei Zhang. <br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2105.09188)] Cited:`9`

### Attribute Editing
**Deep Identity-Aware Transfer of Facial Attributes**<br>
*Mu Li, Wangmeng Zuo, David Zhang*<br>
arxiv 2016. [[PDF](https://arxiv.org/abs/1610.05586)] Cited:`120`

### Others
**Unsupervised Image-to-Image Translation with Generative Prior**<br>
Shuai Yang, Liming Jiang, Ziwei Liu and Chen Change Loy. <br>
CVPR 2022.

### Various Applications
**Sketch Your Own GAN**<br>
Sheng-Yu Wang, David Bau, Jun-Yan Zhu<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2108.02774)] Cited:`9`

### Super-resolution

### Example based image translation
**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**<br>
Shuai Yang, Liming Jiang, Ziwei Liu and Chen Change Loy<br>
CVPR 2022. [[Pytorch](https://github.com/williamyang1991/DualStyleGAN)]

## Unsupervised Image Transaltion
### Swapping Based
**High-Resolution Daytime Translation Without Domain Labels**<br>
I. Anokhin, P. Solovev, D. Korzhenkov, A. Kharlamov, T. Khakhulin, A. Silvestrov, S. Nikolenko, V. Lempitsky, and G. Sterkin.<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2003.08791)] Cited:`35`

**Information Bottleneck Disentanglement for Identity Swapping**<br>
Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, Ran He<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Information_Bottleneck_Disentanglement_for_Identity_Swapping_CVPR_2021_paper.html)]

**Swapping Autoencoder for Deep Image Manipulation**<br>
Taesung Park, Jun-Yan Zhu, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei A. Efros, Richard Zhang<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.00653)] Cited:`97`

**L2M-GAN: Learning to Manipulate Latent Space Semantics for Facial Attribute Editing**<br>
Guoxing Yang, Nanyi Fei, Mingyu Ding, Guangzhen Liu, Zhiwu Lu, Tao Xiang<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_L2M-GAN_Learning_To_Manipulate_Latent_Space_Semantics_for_Facial_Attribute_CVPR_2021_paper.html)]


### Cycle-Consistency Based
**Coupled Generative Adversarial Networks**<br>
Ming-Yu Liu, Oncel Tuzel.<br>
NeurIPS 2016 [[PDF](http://arxiv.org/abs/1606.07536)]

`UNIT` **Unsupervised Image-to-Image Translation Networks.**<br>
Ming-Yu Liu,Thomas Breuel,Jan Kautz<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1703.00848)] Cited:`1725`

`DiscoGAN` **Learning to Discover Cross-Domain Relations with Generative Adversarial Networks**<br>
Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim.<br>
ICML 2017. [[PDF](https://arxiv.org/abs/1703.05192)] Cited:`1338`

`BicycleGAN` **Toward Multimodal Image-to-Image Translation**<br>
Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, Oliver Wang, Eli Shechtman.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1711.11586)] Cited:`883`

`MUNIT` **Multimodal Unsupervised Image-to-Image Translation**<br>
Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1804.04732)] Cited:`1409`

`DRIT` **Diverse Image-to-Image Translation via Disentangled Representations**<br>
Hsin-Ying Lee, Hung-Yu Tseng, Jia-Bin Huang, Maneesh Kumar Singh, Ming-Hsuan Yang.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1808.00948)] Cited:`771`

**Augmented cyclegan: Learning many-to-many mappings from unpaired data.**
*Amjad Almahairi, Sai Rajeswar, Alessandro Sordoni, Philip Bachman, Aaron Courville.*<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1802.10151)] Cited:`283`

**MISO: Mutual Information Loss with Stochastic Style Representations for Multimodal Image-to-Image Translation.**<br>
*Sanghyeon Na, Seungjoo Yoo, Jaegul Choo.*<br>
BMVC 2020. [[PDF](https://arxiv.org/abs/1902.03938)] Cited:`10`

`MSGAN` **Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis**<br>
Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, Ming-Hsuan Yang.<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.05628)] Cited:`192`

`U-GAT-IT` **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee<br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1907.10830)] Cited:`166`

`UVC-GAN` **UVCGAN: UNet Vision Transformer cycle-consistent GAN for unpaired image-to-image translation**<br>
*Dmitrii Torbunov, Yi Huang, Haiwang Yu, Jin Huang, Shinjae Yoo, Meifeng Lin, Brett Viren, Yihui Ren*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.02557)] Cited:`1`


### Beyond Cycle-consistency
`Council-GAN` **Breaking the Cycle - Colleagues are all you need**<br>
*Ori Nizan , Ayellet Tal*<br>
CVPR 2020. [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nizan_Breaking_the_Cycle_-_Colleagues_Are_All_You_Need_CVPR_2020_paper.pdf)]

`ACL-GAN` **Unpaired Image-to-Image Translation using Adversarial Consistency Loss**<br>
Yihao Zhao, Ruihai Wu, Hao Dong.<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2003.04858)] Cited:`23`

### Multi-domain
`StarGAN` **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation**<br>
*Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo*<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.09020)] Cited:`1982`

`DRIT++` **DRIT++: Diverse Image-to-Image Translation via Disentangled Representations**<br>
Hsin-Ying Lee, Hung-Yu Tseng, Qi Mao, Jia-Bin Huang, Yu-Ding Lu, Maneesh Singh, Ming-Hsuan Yang.<br>
IJCV 2019. [[PDF](https://arxiv.org/abs/1905.01270)] Cited:`206`

`StarGANv2` **StarGAN v2: Diverse Image Synthesis for Multiple Domains**<br>
*Yunjey Choi, Youngjung Uh, Jaejun Yoo, Jung-Woo Ha*<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.01865)] Cited:`440`

**Smoothing the Disentangled Latent Style Space for Unsupervised Image-to-Image Translation**<br>
*Yahui Liu, Enver Sangineto, Yajing Chen, Linchao Bao, Haoxian Zhang, Nicu Sebe, Bruno Lepri, Wei Wang, Marco De Nadai*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2106.09016)] Cited:`6`

## Few-shot Image Translation
`FUNIT` **Few-shot unsupervised image-to-image translation.**

Coco-funit:Few-shot unsupervised image translation with a content conditioned style encoder.

## Style Transfer

`WCT` **Universal Style Transfer via Feature Transforms**<br>
Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, Ming-Hsuan Yang. <br>
[[PDF](https://arxiv.org/abs/1705.08086)] Cited:`463`

Style transfer by relaxed optimal transport and self-similarity.


## Others
`GANgealing`**GAN-Supervised Dense Visual Alignment**<br>
William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, Eli Shechtman.<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.05143)] Cited:`5`

# Text-to-Image Synthesis
## End-to-end Training Based
**Generating images from captions with attention.**<br>

`StackGAN` **StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks**<br>
*Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas.*<br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1612.03242)] Cited:`1782`

`StackGAN++` **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks**<br>
*Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas*<br>
TPAMI 2018. [[PDF](https://arxiv.org/abs/1710.10916)] Cited:`581`

`AttnGAN` **AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks**
*Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He.*<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.10485)] Cited:`698`

`DM-GAN` **DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis**<br>
*Minfeng Zhu, Pingbo Pan, Wei Chen, Yi Yang*<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1904.01310)] Cited:`155`

`SD-GAN` **Semantics Disentangling for Text-to-Image Generation**<br>
*Guojun Yin, Bin Liu, Lu Sheng, Nenghai Yu, Xiaogang Wang, Jing Shao*<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1904.01480)] Cited:`78`

`DF-GAN` **A Simple and Effective Baseline for Text-to-Image Synthesis**<br>
*Ming Tao, Hao Tang, Fei Wu, Xiaoyuan Jing, Bingkun Bao, Changsheng Xu.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2008.05865)] Cited:`0`

**Text to Image Generation with Semantic-Spatial Aware GAN**<br>
*Kai Hu, Wentong Liao, Michael Ying Yang, Bodo Rosenhahn*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2104.00567)] Cited:`2`

## Multimodal Pretraining Based
**FuseDream: Training-Free Text-to-Image Generationwith Improved CLIP+GAN Space Optimization**<br>
Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, Qiang Liu<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.01573)] Cited:`5`

`DALLE` **Zero-Shot Text-to-Image Generation**<br>
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever.<br>
ICML 2021. [[PDF](https://arxiv.org/abs/2102.12092)] Cited:`299`

`GLIDE` **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**<br>
*Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen*<br>
arxiv 2021.  [[PDF](https://arxiv.org/pdf/2112.10741.pdf)] [[Pytorch](https://github.com/openai/glide-text2im)]

`DALLE2` **Hierarchical Text-Conditional Image Generation with CLIP Latents**<br>
*Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen*<br>
OpenAI 2022. [[PDF](https://cdn.openai.com/papers/dall-e-2.pdf)]

# Others

## Single Image Generation

`DIP` **Deep Image Prior**<br>
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. <br>
CVPR 2018 [[PDF](https://arxiv.org/abs/1711.10925)] [[Project](https://dmitryulyanov.github.io/deep_image_prior)] Cited:`1257`

`SinGAN` **SinGAN: Learning a Generative Model from a Single Natural Image**<br>
Tamar Rott Shaham, Tali Dekel, Tomer Michaeli. <br>
ICCV 2019 Best Paper. [[PDF](https://arxiv.org/abs/1905.01164)] [[Project](https://tamarott.github.io/SinGAN.htm)] Cited:`349`

`TuiGAN` **TuiGAN: Learning Versatile Image-to-Image Translation with Two Unpaired Images** <br>
Jianxin Lin, Yingxue Pang, Yingce Xia, Zhibo Chen, Jiebo Luo. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.04634)] Cited:`22`

`DeepSIM` **Image Shape Manipulation from a Single Augmented Training Sample**<br>
Yael Vinker, Eliahu Horwitz, Nir Zabari , Yedid Hoshen. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2007.01289)] [[Project](https://www.vision.huji.ac.il/deepsim/)] [[Pytorch](https://github.com/eliahuhorwitz/DeepSIM)] Cited:`1`

# Semi-supervised Learning with GAN
`SemanticGAN` **Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization**<br>
Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.05833)] Cited:`23`

`DatasetGAN` **DatasetGAN: Efﬁcient Labeled Data Factory with Minimal Human Effort**
Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.06490)] Cited:`38`


# Miscellaneous
`SemanticStyleGAN` **SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**<br>
Yichun Shi, Xiao Yang, Yangyue Wan, Xiaohui Shen.<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.02236)] Cited:`0`

**Synthesizing the preferred inputs for neurons in neural networks via deep generator networks.**<br>

**Generating Images with Perceptual Similarity Metrics based on Deep Networks.**<br>
