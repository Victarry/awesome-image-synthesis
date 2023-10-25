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
NeurIPS 2014. [[PDF](https://arxiv.org/abs/1406.2661)] [[Tutorial](https://arxiv.org/abs/1701.00160)] Cited:`396`

`DCGAN` **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** <br>
Alec Radford, Luke Metz, Soumith Chintala. <br>
ICLR 2016. [[PDF](https://arxiv.org/abs/1511.06434)] Cited:`12187`

`PG-GAN` **Progressive Growing of GANs for Improved Quality, Stability, and Variation.** <br>
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1710.10196)] Cited:`5656`

`StyleGAN` **A Style-Based Generator Architecture for Generative Adversarial Networks.** <br>
Tero Karras, Samuli Laine, Timo Aila. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1812.04948)] Cited:`6793`

`BigGAN` **Large Scale GAN Training for High Fidelity Natural Image Synthesis.** <br>
Andrew Brock, Jeff Donahue, Karen Simonyan. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1809.11096)] Cited:`4080`

`StyleGAN2` **Analyzing and Improving the Image Quality of StyleGAN.** <br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.04958)] Cited:`3733`

`VQGAN` **Taming Transformers for High-Resolution Image Synthesis**<br>
Patrick Esser, Robin Rombach, Björn Ommer.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.09841)] [[Project](https://compvis.github.io/taming-transformers/)] Cited:`1149`

`TransGAN` **TransGAN: Two Transformers Can Make One Strong GAN, and That Can Scale Up**<br>
Yifan Jiang, Shiyu Chang, Zhangyang Wang.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2102.07074)] [[Pytorch](https://github.com/asarigun/TransGAN)] Cited:`223`

`StyleGAN3` **Alias-Free Generative Adversarial Networks.**<br>
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2106.12423)] [[Project](https://nvlabs.github.io/stylegan3/)] Cited:`869`

**StyleSwin: Transformer-based GAN for High-resolution Image Generation**<br>
Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, Baining Guo. <br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.10762)] Cited:`90`

**StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets**<br>
Axel Sauer, Katja Schwarz, Andreas Geiger<br>
SIGGRAPH 2022. [[PDF](https://arxiv.org/abs/2202.00273)] Cited:`203`
<details>
<summary>summary</summary>
1. Propose architecture changes based on Projected GANs. (1) Regularization: Only appying path length regularization after model has been sufficiently trained. Blur all images with gaussian filter for the first 200k images. (2) Reduce latent code z dimension to 64 and preserve w code to 512-d. (3) Use pretrained class embdding as conditioning and set it learnable.
2. Design a progressive growing strategy to StyleGAN3.
3. Leverage classifier guidance.
</details>


## GAN Objective
**A Large-Scale Study on Regularization and Normalization in GANs**<br>
*Karol Kurach, Mario Lucic, Xiaohua Zhai, Marcin Michalski, Sylvain Gelly* <br>
ICML 2019. [[PDF](https://arxiv.org/abs/1807.04720)] Cited:`139`

`EB-GAN` **Energy-based Generative Adversarial Networks**<br>
Junbo Zhao, Michael Mathieu, Yann LeCun.<br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1609.03126)] Cited:`1000`

**Towards Principled Methods for Training Generative Adversarial Networks**<br>
Martin Arjovsky, Léon Bottou<br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1701.04862)] Cited:`1794`

<!-- https://towardsdatascience.com/gan-objective-functions-gans-and-their-variations-ad77340bce3c -->
`LSGAN` **Least Squares Generative Adversarial Networks.**<br>
Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley. <br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1611.04076)] Cited:`3846`

`WGAN` **Wasserstein GAN**<br>
Martin Arjovsky, Soumith Chintala, Léon Bottou.<br>
ICML 2017. [[PDF](https://arxiv.org/abs/1701.07875)] Cited:`3748`

`GGAN` **Geometric GAN** <br>
Jae Hyun Lim, Jong Chul Ye. <br>
Axiv 2017. [[PDF](https://arxiv.org/abs/1705.02894)] Cited:`357`

`AC-GAN` **Conditional Image Synthesis With Auxiliary Classifier GANs** <br>
Augustus Odena, Christopher Olah, Jonathon Shlens. <br>
ICML 2017. [[PDF](https://arxiv.org/abs/1610.09585)] Cited:`2716`

**cGANs with Projection Discriminator**<br>
Takeru Miyato, Masanori Koyama. <br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1802.05637)] Cited:`671`

`S³-GAN` **High-Fidelity Image Generation With Fewer Labels**<br>
Mario Lucic*, Michael Tschannen*, Marvin Ritter*, Xiaohua Zhai, Olivier Bachem, Sylvain Gelly. <br>
ICML 2019. [[PDF](https://arxiv.org/abs/1903.02271)] [[Tensorflow](https://github.com/google/compare_gan)] Cited:`133`


## Autoencoder-based framework
`VAE` **Auto-Encoding Variational Bayes.**<br>
Diederik P.Kingma, Max Welling.<br>
ICLR 2014. [[PDF](https://arxiv.org/abs/1312.6114)] Cited:`21261`

`AAE` **Adversarial Autoencoders.**<br>
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey.<br>
arxiv 2015. [[PDF](https://arxiv.org/abs/1511.05644)] Cited:`1816`

`VAE/GAN` **Autoencoding beyond pixels using a learned similarity metric.** <br>
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther.<br>
ICML 2016. [[PDF](https://arxiv.org/abs/1512.09300)] Cited:`1763`

`VampPrior` **VAE with a VampPrior** <br>
Jakub M. Tomczak, Max Welling.<br>
AISTATS 2018. [[PDF](https://arxiv.org/abs/1705.07120)] [[Pytorch](https://github.com/jmtomczak/vae_vampprior)] Cited:`518`

`BiGAN` **Adversarial Feature Learning**<br>
Jeff Donahue, Philipp Krähenbühl, Trevor Darrell. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1605.09782)] Cited:`1640`

`AIL` **Adversarial Learned Inference**<br>
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1606.00704)] Cited:`1204`

`VEEGAN` **Veegan: Reducing mode collapse in gans using implicit variational learning.**<br>
Akash Srivastava, Lazar Valkov, Chris Russell, Michael U. Gutmann, Charles Sutton.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1705.07761)] [[Github](https://github.com/akashgit/VEEGAN)] Cited:`554`

`AGE` **Adversarial Generator-Encoder Networks.**<br>
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky.<br>
AAAI 2018. [[PDF](https://arxiv.org/abs/1704.02304)] [[Pytorch](https://github.com/DmitryUlyanov/AGE)] Cited:`122`

`IntroVAE` **IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis.**<br>
Huaibo Huang, Zhihang Li, Ran He, Zhenan Sun, Tieniu Tan. <br>
NeurIPS 2018. [[PDF](https://arxiv.org/abs/1807.06358)] Cited:`206`

**Disentangled Inference for {GAN}s with Latently Invertible Autoencoder**<br>
*Jiapeng Zhu, Deli Zhao, Bo Zhang, Bolei Zhou*<br>
IJCV 2020. [[PDF](https://arxiv.org/abs/1906.08090)] Cited:`23`

`ALAE` **Adversarial Latent Autoencoders**<br>
Stanislav Pidhorskyi, Donald Adjeroh, Gianfranco Doretto. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2004.04467)] Cited:`212`

## VAEs
**Variational Inference with Normalizing Flows**<br>
*Danilo Jimenez Rezende, Shakir Mohamed*<br>
ICML 2015. [[PDF](https://arxiv.org/abs/1505.05770)] Cited:`3025`

**Improved Variational Inference with Inverse Autoregressive Flow**<br>
*Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling*<br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.04934)] Cited:`1525`

**NVAE: A Deep Hierarchical Variational Autoencoder**<br>
*Arash Vahdat, Jan Kautz*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.03898)] Cited:`570`

## Diffusion Models
**Improved techniques for training score-based generative models.**<br>
_Yang Song, Stefano Ermon_<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2006.09011)] Cited:`569`

`DDPM` **Denoising Diffusion Probabilistic Models**<br>
_Jonathan Ho, Ajay Jain, Pieter Abbeel_<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2006.11239)] Cited:`4185`

**Score-based generative modeling through stochastic differential equations**<br>
_Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole_<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2011.13456)] Cited:`1869`

`Improved-DDPM` **Improved Denoising Diffusion Probabilistic Models**<br>
_Alex Nichol, Prafulla Dhariwal_<br>
ICML 2021. [[PDF](https://arxiv.org/abs/2102.09672)] Cited:`1232`

**Variational Diffusion Models.**<br>
_Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho_<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2107.00630)] Cited:`422`

`Guided-Diffusion` **Diffusion Models Beat GANs on Image Synthesis**<br>
_Prafulla Dhariwal, Alex Nichol_<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2105.05233)] Cited:`2259`

**Classifier-Free Diffusion Guidance.**<br>
_Jonathan Ho, Tim Salimans_<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2207.12598)] Cited:`941`

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations**<br>
_Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2108.01073)] Cited:`348`

**DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models**<br>
_Gwanghyun Kim, Taesung Kwon, Jong Chul Ye_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2110.02711)] Cited:`223`

**Blended Diffusion: Text-driven Editing of Natural Images**<br>
_Omri Avrahami, Dani Lischinski, Ohad Fried_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.14818)] Cited:`287`

**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**<br>
_Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen_<br>
ICML 2022. [[PDF](https://arxiv.org/abs/2112.10741)] Cited:`1254`

**Palette: Image-to-Image diffusion models.**<br>
_Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, Mohammad Norouzi_<br>
SIGGRAPH 2022. [[PDF](https://arxiv.org/abs/2111.05826)] Cited:`549`

**RePaint: Inpainting using Denoising Diffusion Probabilistic Models**<br>
_Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, Luc Van Gool_<br>
CVPR 2022 [[PDF](https://arxiv.org/abs/2201.09865)] Cited:`431`

## Disentangled Image Generation
`DC-IGN` **Deep Convolutional Inverse Graphics Network**<br>
Tejas D. Kulkarni, Will Whitney, Pushmeet Kohli, Joshua B. Tenenbaum. <br>
NeurIPS 2015. [[PDF](Deep Convolutional Inverse Graphics Network)]

`InfoGAN` **InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets**<br>
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel. <br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.03657)] Cited:`3717`

`Beta-VAE` **beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**<br>
I. Higgins, L. Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, M. Botvinick, S. Mohamed, Alexander Lerchner. <br>
ICLR 2017. [[PDF](https://openreview.net/forum?id=Sy2fzU9gl)]

`AnnealedVAE` **Understanding disentangling in β-VAE**<br>
Christopher P. Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters, Guillaume Desjardins, Alexander Lerchner. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1804.03599)] Cited:`693`

`Factor-VAE` **Disentangling by Factorising**<br>
Hyunjik Kim, Andriy Mnih. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1802.05983)] Cited:`1091`

`DCI` **A framework for the quantitative evaluation of disentangled representations.**<br>
*Cian Eastwood, Christopher K. I. Williams*
ICLR 2018. [[PDF](https://openreview.net/pdf?id=By-7dz-AZ)] 

**Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.**<br>
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem.<br>
ICML(best paper award) 2019. [[PDF](https://arxiv.org/abs/1811.12359)] Cited:`1113`

## Regularization / Limited Data
`WGAN-GP` **Improved training of wasserstein gans**<br>
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1704.00028)] Cited:`7630`

**The Numerics of GANs**<br>
*Lars Mescheder, Sebastian Nowozin, Andreas Geiger*<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1705.10461)] Cited:`409`

`R1-regularization` **Which Training Methods for GANs do actually Converge?**<br>
*Lars Mescheder, Andreas Geiger, Sebastian Nowozin.*<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1801.04406)] Cited:`1203`

`SN-GAN` **Spectral Normalization for Generative Adversarial Networks.**<br>
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1802.05957)] Cited:`3669`

`CR-GAN` **Consistency regularization for generative adversarial networks.**<br>
Han Zhang, Zizhao Zhang, Augustus Odena, Honglak Lee. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1910.12027)] Cited:`218`
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
NeurIPS 2020.  [[PDF](https://arxiv.org/abs/2006.10738)] [[Project](https://hanlab.mit.edu/projects/data-efficient-gans/)]<br> Cited:`419`

`ICR-GAN` **Improved consistency regularization for GANs.** <br>
Zhengli Zhao, Sameer Singh, Honglak Lee, Zizhao Zhang, Augustus Odena, Han Zhang. <br>
AAAI 2021. [[PDF](https://arxiv.org/abs/2002.04724)] Cited:`107`
<details>
<summary>Summary</summary>
Motivation: The consistency regularization will introduce artifacts into GANs sample correponding to <br>
Method: 1. (bCR) In addition to CR,  bCR also encourage discriminator output the same feature for generated image and its augmentation. 
2. (zCR) zCR encourage discriminator insensitive to generated images with perturbed latent code, while encourage generator sensitive to that. <br>
Experiment: the augmentation to image is same as CR-GAN, the augmentation to latent vector is guassian noise.
</details>

`StyleGAN-ADA` **Training Generative Adversarial Networks with Limited Data.**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila. <br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2006.06676)] [[Tensorflow](https://github.com/NVlabs/stylegan2-ada)] [[Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)] Cited:`1210`
<!-- <details>
<summary>summary</summary>
Motivation: <br>
1. 如何防止augmentation leak.
2. 数据少的时候, discriminator会过拟合, 表现在很容易区分出real和fake, 而FID很早就收敛,然后开始变差
</details> -->

**Gradient Normalization for Generative Adversarial Networks.**<br>
Yi-Lun Wu, Hong-Han Shuai, Zhi-Rui Tam, Hong-Yu Chiu. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.02235)] Cited:`36`

**Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data.**<br>
Liming Jiang, Bo Dai, Wayne Wu, Chen Change Loy. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2111.06849)] Cited:`55`

## Metric
`Inception-Score/IS` **Improved Techniques for Training GANs**
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. <br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.03498)] Cited:`7186`

`FID, TTUR` **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium**<br>
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. <br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1706.08500)] Cited:`331`

`SWD` **Sliced Wasserstein Generative Models**
Jiqing Wu, Zhiwu Huang, Dinesh Acharya, Wen Li, Janine Thoma, Danda Pani Paudel, Luc Van Gool. <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1706.02631)] Cited:`0`

## Fast Convergence
`FastGAN` **Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis**<br>
Bingchen Liu, Yizhe Zhu, Kunpeng Song, Ahmed Elgammal. <br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.04775)] Cited:`153`

`ProjectedGAN` **Projected GANs Converge Faster**<br>
Axel Sauer, Kashyap Chitta, Jens Müller, Andreas Geiger<br>
[[PDF](https://arxiv.org/abs/2111.01007)] [[Project](https://sites.google.com/view/projected-gan/)] [[Pytorch](https://github.com/autonomousvision/projected_gan)] Cited:`120`

## GAN Adaptation
**Transferring GANs: generating images from limited data.**<br>
Yaxing Wang, Chenshen Wu, Luis Herranz, Joost van de Weijer, Abel Gonzalez-Garcia, Bogdan Raducanu.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1805.01677)] Cited:`224`

**Image Generation From Small Datasets via Batch Statistics Adaptation.**<br>
Atsuhiro Noguchi, Tatsuya Harada.<br>
ICCV 2019 [[PDF](https://arxiv.org/abs/1904.01774)] Cited:`154`

**Freeze Discriminator: A Simple Baseline for Fine-tuning GANs.**<br>
Sangwoo Mo, Minsu Cho, Jinwoo Shin.<br>
CVPRW 2020 [[PDF](https://arxiv.org/abs/2002.10964)] [[Pytorch](https://github.com/sangwoomo/FreezeD)] Cited:`164`

**Resolution dependant GAN interpolation for controllable image synthesis between domains.**<br>
Justin N. M. Pinkney, Doron Adler<br>
NeruIPS workshop 2020. [[PDF](https://arxiv.org/abs/2010.05334)] Cited:`102`

**Few-shot image generation with elastic weight consolidation.**
Yijun Li, Richard Zhang, Jingwan Lu, Eli Shechtman<br>
NeruIPS 2020. [[PDF](https://arxiv.org/abs/2012.02780)] Cited:`116`

**Minegan: effective knowledge transfer from gans to target domains with few images.**
*Yaxing Wang, Abel Gonzalez-Garcia, David Berga, Luis Herranz, Fahad Shahbaz Khan, Joost van de Weijer*<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.05270)] Cited:`147`

**One-Shot Domain Adaptation For Face Generation**<br>
_Chao Yang, Ser-Nam Lim_<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2003.12869)] Cited:`31`

**Unsupervised image-to-image translation via pre-trained StyleGAN2 network**<br>
_Jialu Huang, Jing Liao, Sam Kwong_<br>
TMM 2021. [[PDF](https://arxiv.org/abs/2010.05713)] Cited:`40`

**Few-shot Adaptation of Generative Adversarial Networks**<br>
Esther Robb, Wen-Sheng Chu, Abhishek Kumar, Jia-Bin Huang.<br>
arxiv 2020 [[PDF](https://arxiv.org/abs/2010.11943)] Cited:`68`

**AgileGAN: stylizing portraits by inversion-consistent transfer learning.**<br>
_Guoxian Song, Linjie Luo, Jing Liu, Wan-Chun Ma, Chunpong Lai, Chuanxia Zheng, Tat-Jen Cham_<br>
TOG/SIGGRAPH 2021. [[PDF](https://guoxiansong.github.io/homepage/paper/AgileGAN.pdf)] [[Project](https://guoxiansong.github.io/homepage/agilegan.html)]

**Few-shot Image Generation via Cross-domain Correspondence**<br>
Utkarsh Ojha, Yijun Li, Jingwan Lu, Alexei A. Efros, Yong Jae Lee, Eli Shechtman, Richard Zhang.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.06820)] Cited:`159`

**StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators**<br>
Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, Daniel Cohen-Or.<br>
arxiv 2021 [[PDF](https://arxiv.org/abs/2108.00946)] [[Project](https://stylegan-nada.github.io/)] Cited:`244`

**Stylealign: Analysis and Applications of Aligned StyleGAN Models**<br>
_Zongze Wu, Yotam Nitzan, Eli Shechtman, Dani Lischinski_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2110.11323)] Cited:`39`

**One-Shot Generative Domain Adaptation**<br>
_Ceyuan Yang, Yujun Shen*, Zhiyi Zhang, Yinghao Xu, Jiapeng Zhu, Zhirong Wu, Bolei Zhou_ <br>
arXiv 2021. [[PDF](https://arxiv.org/abs/2111.09876)] Cited:`29`

**Mind the Gap: Domain Gap Control for Single Shot Domain Adaptation for Generative Adversarial Networks**<br>
_Peihao Zhu, Rameen Abdal, John Femiani, Peter Wonka_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2110.08398)] Cited:`55`

**Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment**<br>
_Jiayu Xiao, Liang Li, Chaofei Wang, Zheng-Jun Zha, Qingming Huang_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.04121)] Cited:`35`

**JoJoGAN: One Shot Face Stylization**<br>
_Min Jin Chong, David Forsyth_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2112.11641)] Cited:`44`

**When why and which pretrained GANs are useful?**<br>
*Timofey Grigoryev, Andrey Voynov, Artem Babenko*<br>
ICLR 2022. [[PDF](https://openreview.net/forum?id=4Ycr8oeCoIh)]

**CtlGAN: Few-shot Artistic Portraits Generation with Contrastive Transfer Learning**<br>
_Yue Wang, Ran Yi, Ying Tai, Chengjie Wang, and Lizhuang Ma_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.08612)] Cited:`9`

**One-Shot Adaptation of GAN in Just One CLIP**<br>
_Gihyun Kwon, Jong Chul Ye_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.09301)] Cited:`18`

**A Closer Look at Few-shot Image Generation**<br>
_Yunqing Zhao, Henghui Ding, Houjing Huang, Ngai-Man Cheung_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2205.03805)] Cited:`35`

**Diffusion Guided Domain Adaptation of Image Generators**<br>
_Kunpeng Song, Ligong Han, Bingchen Liu, Dimitris Metaxas, Ahmed Elgammal_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2212.04473)] Cited:`11`

**Domain Expansion of Image Generators**<br>
_Yotam Nitzan, Michaël Gharbi, Richard Zhang, Taesung Park, Jun-Yan Zhu, Daniel Cohen-Or, Eli Shechtman_<br>
arxiv 2023. [[PDF](https://arxiv.org/abs/2301.05225)] Cited:`4`

## VAEs
**Variational Inference with Normalizing Flows**<br>
*Danilo Jimenez Rezende, Shakir Mohamed*<br>
ICML 2015. [[PDF](https://arxiv.org/abs/1505.05770)] Cited:`3025`

**Improved Variational Inference with Inverse Autoregressive Flow**<br>
*Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling*<br>
NeurIPS 2016. [[PDF](https://arxiv.org/abs/1606.04934)] Cited:`1525`

**NVAE: A Deep Hierarchical Variational Autoencoder**<br>
*Arash Vahdat, Jan Kautz*<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.03898)] Cited:`570`

## Other Generative Models
**Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space**<br>
_Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, Jason Yosinski_<br>
CVPR 2017. [[PDF](https://arxiv.org/abs/1612.00005)] Cited:`607`

`GLO` **Optimizing the Latent Space of Generative Networks**<br>
Piotr Bojanowski, Armand Joulin, David Lopez-Paz, Arthur Szlam<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1707.05776)] Cited:`351`

**Non-Adversarial Image Synthesis with Generative Latent Nearest Neighbors**<br>
_Yedid Hoshen, Jitendra Malik_<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1812.08985)] Cited:`49`

## Latent Interpolation
**Sampling generative networks: Notes on a few effective techniques.**<br>
Tom White.<br>
arxiv 2016 [[PDF](https://arxiv.org/abs/1609.04468v2)] Cited:`114`

**Latent space oddity: on the curvature of deep generative models**<br>
Georgios Arvanitidis, Lars Kai Hansen, Søren Hauberg.<br>
ICLR 2018. [[PDF](https://arxiv.org/abs/1710.11379)] Cited:`206`

**Feature-Based Metrics for Exploring the Latent Space of Generative Models**<br>
Samuli Laine.<br>
ICLR 2018 Workshop. [[PDF](https://openreview.net/forum?id=BJslDBkwG)]

# Two-stage Generation Models
`VQ-VAE` **Neural Discrete Representation Learning**<br>
_Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu_<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1711.00937)] Cited:`2470`

`VQ-VAE-2` **Generating Diverse High-Fidelity Images with VQ-VAE-2**<br>
_Ali Razavi, Aaron van den Oord, Oriol Vinyals_<br>
NeurIPS 2019. [[PDF](https://arxiv.org/abs/1906.00446)] Cited:`1091`

`VQGAN` **Taming Transformers for High-Resolution Image Synthesis**<br>
Patrick Esser, Robin Rombach, Björn Ommer.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.09841)] [[Project](https://compvis.github.io/taming-transformers/)] Cited:`1149`

`DALLE` **Zero-Shot Text-to-Image Generation**<br>
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever.<br>
ICML 2021. [[PDF](https://arxiv.org/abs/2102.12092)] Cited:`2223`

**The Image Local Autoregressive Transformer**<br>
_Chenjie Cao, Yuxin Hong, Xiang Li, Chengrong Wang, Chengming Xu, XiangYang Xue, Yanwei Fu_<br>
NeruIPS 2021. [[PDF](https://arxiv.org/abs/2106.02514)] Cited:`9`

`MaskGIT` **MaskGIT: Masked Generative Image Transformer**<br>
_Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2202.04200)] Cited:`191`

`VQGAN-CLIP` **VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance**<br>
_Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, Edward Raff_<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2204.08583)] Cited:`193`

`ASSET` **Autoregressive Semantic Scene Editing with Transformers at High Resolutions**<br>
_Difan Liu, Sandesh Shetty, Tobias Hinz, Matthew Fisher, Richard Zhang, Taesung Park, Evangelos Kalogerakis_<br>
SIGGRAPH 2022. [[Pytorch](https://github.com/DifanLiu/ASSET)] 

`CLIP-GEN` **CLIP-GEN: Language-Free Training of a Text-to-Image Generator with CLIP**<br>
_Zihao Wang, Wei Liu, Qian He, Xinglong Wu, Zili Yi_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.00386)] Cited:`39`

`PUT` **Reduce Information Loss in Transformers for Pluralistic Image Inpainting**<br>
_Qiankun Liu, Zhentao Tan, Dongdong Chen, Qi Chu, Xiyang Dai, Yinpeng Chen, Mengchen Liu, Lu Yuan, Nenghai Yu_<br>
CVPR 2022. [[Pytorch](https://github.com/liuqk3/PUT)]

**High-Quality Pluralistic Image Completion via Code Shared VQGAN**<br>
_Chuanxia Zheng, Guoxian Song, Tat-Jen Cham, Jianfei Cai, Dinh Phung, Linjie Luo_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2204.01931)] Cited:`5`

**L-Verse: Bidirectional Generation Between Image and Text**<br>
_Taehoon Kim, Gwangmo Song, Sihaeng Lee, Sangyun Kim, Yewon Seo, Soonyoung Lee, Seung Hwan Kim, Honglak Lee, Kyunghoon Bae_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.11133)] Cited:`13`

**Unleashing Transformers: Parallel Token Prediction with Discrete Absorbing Diffusion for Fast High-Resolution Image Generation from Vector-Quantized Codes.**<br>
_Sam Bond-Taylor, Peter Hessey, Hiroshi Sasaki, Toby P. Breckon, Chris G. Willcocks_<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2111.12701)] Cited:`35`

`MaskGIT` **MaskGIT: Masked Generative Image Transformer**<br>
_Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2202.04200)] Cited:`191`

**Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis.**<br>
_Patrick Esser, Robin Rombach, Andreas Blattmann, Björn Ommer_<br>
NeruIPS 2021. [[PDF](https://arxiv.org/abs/2108.08827)] Cited:`91`

**Vector Quantized Diffusion Model for Text-to-Image Synthesis**<br>
_Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.14822)] Cited:`304`

**Improved Vector Quantized Diffusion Models**<br>
_Zhicong Tang, Shuyang Gu, Jianmin Bao, Dong Chen, Fang Wen_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2205.16007)] Cited:`39`

`Text2Human` **Text2Human: Text-Driven Controllable Human Image Generation**<br>
_Yuming Jiang, Shuai Yang, Haonan Qiu, Wayne Wu, Chen Change Loy, Ziwei Liu_<br>
SIGGRAPH 2022. [[PDF](https://arxiv.org/abs/2205.15996)] Cited:`39`

`RQ-VAE` **Autoregressive image generation using residual quantization**<br>
_Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, Wook-Shin Han_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.01941)] Cited:`76`

**Draft-and-Revise: Effective Image Generation with Contextual RQ-Transformer**<br>



# Image Manipulation with Deep Generative Model
## GAN Inversion

`iGAN` **Generative Visual Manipulation on the Natural Image Manifold**<br>
Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros. <br>
ECCV 2016. [[PDF](https://arxiv.org/abs/1609.03552)] [[github](https://github.com/junyanz/iGAN)] Cited:`1268`

`IcGAN` **Invertible Conditional GANs for image editing**<br>
Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, Jose M. Álvarez<br>
NIPS 2016 Workshop. [[PDF](https://arxiv.org/abs/1611.06355)] Cited:`591`

**Neural photo editing with introspective adversarial networks**<br>
Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston. <br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1609.07093)] Cited:`412`

**Inverting The Generator of A Generative Adversarial Network.**<br>
Antonia Creswell, Anil Anthony Bharath. <br>
NeurIPS 2016 Workshop. [[PDF](https://arxiv.org/abs/1611.05644)] Cited:`289`

`GAN Paint` **Semantic Photo Manipulation with a Generative Image Prior**<br>
David Bau, Hendrik Strobelt, William Peebles, Jonas Wulff, Bolei Zhou, Jun-Yan Zhu, Antonio Torralba.<br>
SIGGRAPH 2019. [[PDF](https://arxiv.org/abs/2005.07727)] Cited:`302`

`GANSeeing` **Seeing What a GAN Cannot Generate.**<br>
David Bau, Jun-Yan Zhu, Jonas Wulff, William Peebles, Hendrik Strobelt, Bolei Zhou, Antonio Torralba.<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1910.11626)] Cited:`228`
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
ICCV 2019. [[PDF](https://arxiv.org/abs/1904.03189)] Cited:`833`

**Image2StyleGAN++: How to Edit the Embedded Images?**<br>
Rameen Abdal, Yipeng Qin, Peter Wonka. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1911.11544)] Cited:`426`

<!-- mGANPrior -->

`IDInvert` **In-Domain GAN Inversion for Real Image Editing**<br>
Jiapeng Zhu, Yujun Shen, Deli Zhao, Bolei Zhou. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.00049)] Cited:`494`
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
CVPR 2020. [[PDF](https://arxiv.org/abs/2004.14367)] [[Pytorch](https://github.com/cyrilzakka/GANLocalEditing)] Cited:`222`
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
arxiv 2020. [[PDF](https://arxiv.org/abs/2009.06529)] Cited:`38`

**Improved StyleGAN Embedding: Where are the Good Latents?**<br>
Peihao Zhu, Rameen Abdal, Yipeng Qin, John Femiani, Peter Wonka<br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2012.09036)] Cited:`79`

`pix2latent` **Transforming and Projecting Images into Class-conditional Generative Networks**<br>
Minyoung Huh,Richard Zhang,Jun-Yan Zhu,Sylvain Paris,Aaron Hertzmann<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2005.01703)] Cited:`87`

`pSp,pixel2style2pixel` **Encoding in style: a stylegan encoder for image-to-image translation.**<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2008.00951)] [[Pytorch](https://github.com/eladrich/pixel2style2pixel)] Cited:`743`

`e4e, encode for editing` **Designing an encoder for StyleGAN image manipulation.**<br>
Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, Daniel Cohen-Or.<br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2102.02766)] Cited:`456`

`ReStyle` **Restyle: A residual-based stylegan encoder via iterative refinement.**<br>
Yuval Alaluf, Or Patashnik, Daniel Cohen-Or. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.02699)] [[Project](https://yuval-alaluf.github.io/restyle-encoder/)] Cited:`248`


**Collaborative Learning for Faster StyleGAN Embedding.** <br>
Shanyan Guan, Ying Tai, Bingbing Ni, Feida Zhu, Feiyue Huang, Xiaokang Yang. <br>
arxiv 2020. [[PDF](https://arxiv.org/abs/2007.01758)] Cited:`90`
<details>
<summary>Summary</summary>
1. Motivation: Traditional methods either use optimization based of learning based methods to get the embeded latent code. However, the optimization based method suffers from large time cost and is sensitive to initiialization. The learning based method get relative worse image quality due to the lack of direct supervision on latent code. <br>
2. This paper introduce a collaborartive training process consisting of an learnable embedding network and an optimization-based iterator to train the embedding network. For each training batch, the embedding network firstly encode the images as initialization code of the iterator, then the iterator update 100 times to optimize MSE and LPIPS loss of generated images with target image, after that the updated embedding code is used as target signal to train the embedding network with latent code distance, image-level and feature-level loss.<br>
3. The embedding network consists of a pretrained Arcface model as identity encoder, an attribute encoder built with ResBlock, the output identity feature and attribute feature are combined via linear modulation(denomarlization in SPADE). After that a Treeconnect(a sparse alterative to fully-connected layer) is used to output the final embeded code.
</details>

**Pivotal Tuning for Latent-based Editing of Real Images**<br>
Daniel Roich, Ron Mokady, Amit H. Bermano, Daniel Cohen-Or. <br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2106.05744)] Cited:`300`

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing.**<br>
Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, Amit H. Bermano. <br>
CVPR 2022 [[PDF](https://arxiv.org/abs/2111.15666)] [[Project](https://yuval-alaluf.github.io/hyperstyle/)] Cited:`156`

**High-Fidelity GAN Inversion for Image Attribute Editing**<br>
Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, Qifeng Chen. <br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2109.06590)] Cited:`143`

## Supervised GAN Manipulation
**GAN Dissection: Visualizing and Understanding Generative Adversarial Networks**<br>
David Bau, Jun-Yan Zhu, Hendrik Strobelt, Bolei Zhou, Joshua B. Tenenbaum, William T. Freeman, Antonio Torralba. <br>
ICLR 2019. [[PDF](https://arxiv.org/abs/1811.10597)] [[Project](http://gandissect.csail.mit.edu/)]. Cited:`380`

**On the "steerability" of generative adversarial networks.**<br>
Ali Jahanian, Lucy Chai, Phillip Isola. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1907.07171)] [[Project](https://ali-design.github.io/gan_steerability/)] [[Pytorch](https://github.com/ali-design/gan_steerability)] Cited:`348`

**Controlling generative models with continuous factors of variations.**<br>
Antoine Plumerault, Hervé Le Borgne, Céline Hudelot. <br>
ICLR 2020. [[PDF](https://arxiv.org/abs/2001.10238)] Cited:`112`

`InterFaceGAN` **Interpreting the Latent Space of GANs for Semantic Face Editing**<br>
Yujun Shen, Jinjin Gu, Xiaoou Tang, Bolei Zhou. <br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1907.10786)] [[Project](https://genforce.github.io/interfacegan/)] Cited:`831`

**Enjoy your editing: Controllable gans for image editing via latent space navigation**<br>
Peiye Zhuang, Oluwasanmi Koyejo, Alexander G. Schwing<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2102.01187)] Cited:`56`

**Only a matter of style: Age transformation using a style-based regression model.**<br>
Yuval Alaluf, Or Patashnik, Daniel Cohen-Or<br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2102.02754)] Cited:`89`

**Discovering Interpretable Latent Space Directions of GANs Beyond Binary Attributes.**<br>
*Huiting Yang, Liangyu Chai, Qiang Wen, Shuang Zhao, Zixun Sun, Shengfeng He.*<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Discovering_Interpretable_Latent_Space_Directions_of_GANs_Beyond_Binary_Attributes_CVPR_2021_paper.pdf)]

`StyleSpace` **StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation**<br>
Zongze Wu, Dani Lischinski, Eli Shechtman. <br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2011.12799)] Cited:`339`

`StyleFlow` **StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows** <br>
Rameen Abdal, Peihao Zhu, Niloy Mitra, Peter Wonka. <br>
SIGGRAPH 2021. [[PDF](https://arxiv.org/abs/2008.02401)] Cited:`382`

**A Latent Transformer for Disentangled Face Editing in Images and Videos.**<br>
*Xu Yao, Alasdair Newson, Yann Gousseau, Pierre Hellier.*<br>
ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_A_Latent_Transformer_for_Disentangled_Face_Editing_in_Images_and_ICCV_2021_paper.html)] [[ArXiV](https://arxiv.org/abs/2106.11895)] [[Github](https://github.com/InterDigitalInc/latent-transformer)] Cited:`48`

**Controllable and Compositional Generation with Latent-Space Energy-Based Models.**<br>
*Weili Nie, Arash Vahdat, Anima Anandkumar.*<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2110.10873)] Cited:`39`

`EditGAN` **EditGAN: High-Precision Semantic Image Editing**<br>
Huan Ling, Karsten Kreis, Daiqing Li, Seung Wook Kim, Antonio Torralba, Sanja Fidler. <br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2111.03186)] Cited:`129`

`StyleFusion` **StyleFusion: A Generative Model for Disentangling Spatial Segments**<br>
*Omer Kafri, Or Patashnik, Yuval Alaluf, Daniel Cohen-Or*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2107.07437)] Cited:`26`


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
ICML 2020. [[PDF](https://arxiv.org/abs/2002.03754)] Cited:`305`

`GANSpace`**GANSpace: Discovering Interpretable GAN Controls**<br>
Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, Sylvain Paris.<br>
NeurIPS 2020 [[PDF](https://arxiv.org/abs/2004.02546)] [[Pytorch](https://github.com/harskish/ganspace)] Cited:`652`

**The Hessian Penalty: A Weak Prior for Unsupervised Disentanglement**<br>
William Peebles, John Peebles, Jun-Yan Zhu, Alexei Efros, Antonio Torralba<br>
ECCV 2020 [[PDF](https://arxiv.org/abs/2008.10599)] [[Project](https://www.wpeebles.com/hessian-penalty)] Cited:`97`

**The Geometry of Deep Generative Image Models and its Applications**<br>
Binxu Wang, Carlos R. Ponce.<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2101.06006)] Cited:`31`

**GAN Steerability without optimization.**<br>
Nurit Spingarn-Eliezer, Ron Banner, Tomer Michaeli<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2012.05328)] Cited:`49`

`SeFa` **Closed-Form Factorization of Latent Semantics in GANs**<br>
Yujun Shen, Bolei Zhou. <br>
CVPR 2021 [[PDF](https://arxiv.org/abs/2007.06600)] [[Project](https://genforce.github.io/sefa/)] Cited:`436`

`NaviGAN` **Navigating the GAN Parameter Space for Semantic Image Editing**<br>
Anton Cherepkov, Andrey Voynov, Artem Babenko.<br>
CVPR 2021 [[PDF](https://arxiv.org/abs/2011.13786)] [[Pytorch](https://github.com/yandex-research/navigan)] Cited:`50`

**EigenGAN: Layer-Wise Eigen-Learning for GANs.**<br>
*Zhenliang He, Meina Kan, Shiguang Shan.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2104.12476)] [[Github](https://github.com/LynnHo/EigenGAN-Tensorflow)] Cited:`38`

**Toward a Visual Concept Vocabulary for GAN Latent Space.**<br>
*Sarah Schwettmann, Evan Hernandez, David Bau, Samuel Klein, Jacob Andreas, Antonio Torralba*.<br>
ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/html/Schwettmann_Toward_a_Visual_Concept_Vocabulary_for_GAN_Latent_Space_ICCV_2021_paper.html)] [[Project](https://visualvocab.csail.mit.edu/)]

**WarpedGANSpace: Finding Non-linear RBF Paths in GAN Latent Space.**<br>
*Christos Tzelepis, Georgios Tzimiropoulos, Ioannis Patras.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2109.13357)] [[Github](https://github.com/chi0tzp/WarpedGANSpace)] Cited:`44`

**OroJaR: Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation.**<br>
*Yuxiang Wei, Yupeng Shi, Xiao Liu, Zhilong Ji, Yuan Gao, Zhongqin Wu, Wangmeng Zuo.*<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2108.07668)] [[Github](https://github.com/csyxwei/OroJaR)] Cited:`34`

**Optimizing Latent Space Directions For GAN-based Local Image Editing.**<br>
*Ehsan Pajouheshgar, Tong Zhang, Sabine Süsstrunk.*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2111.12583)] [[Pytorch](https://github.com/IVRL/LELSD)] Cited:`6`

**Discovering Density-Preserving Latent Space Walks in GANs for Semantic Image Transformations.**<br>
*Guanyue Li, Yi Liu, Xiwen Wei, Yang Zhang, Si Wu, Yong Xu, Hau San Wong.*<br>
ACM MM 2021. [[PDF](https://dl.acm.org/doi/abs/10.1145/3474085.3475293)]

**Disentangled Representations from Non-Disentangled Models**<br>
*Valentin Khrulkov, Leyla Mirvakhabova, Ivan Oseledets, Artem Babenko*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2102.06204)] Cited:`11`

**Do Not Escape From the Manifold: Discovering the Local Coordinates on the Latent Space of GANs.**<br>
*Jaewoong Choi, Changyeon Yoon, Junho Lee, Jung Ho Park, Geonho Hwang, Myungjoo Kang.*<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2106.06959)] Cited:`15`

`Disco` **Learning Disentangled Representation by Exploiting Pretrained Generative Models: A Contrastive Learning View**<br>
*Xuanchi Ren, Tao Yang, Yuwang Wang, Wenjun Zeng*<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2102.10543)] Cited:`19`

**Rayleigh EigenDirections (REDs): GAN latent space traversals for multidimensional features.**<br>
*Guha Balakrishnan, Raghudeep Gadde, Aleix Martinez, Pietro Perona.*<br>
arxiv 2022. [[PDF](https://arxiv.org/pdf/2201.10423.pdf)]

**Low-Rank Subspaces in GANs**<br>
_Jiapeng Zhu, Ruili Feng, Yujun Shen, Deli Zhao, Zhengjun Zha, Jingren Zhou, Qifeng Chen_<br>
NeurIPS 2021. [[PDF](https://arxiv.org/abs/2106.04488)] Cited:`52`

**Region-Based Semantic Factorization in GANs**<br>
*Jiapeng Zhu, Yujun Shen, Yinghao Xu, Deli Zhao, Qifeng Chen.*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2202.09649)] Cited:`20`

**Fantastic Style Channels and Where to Find Them: A Submodular Framework for Discovering Diverse Directions in GANs**<br>


## CLIP based
**StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**<br>
_Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, Dani Lischinski_<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.17249)] [[Pytorch](https://github.com/orpatashnik/StyleCLIP)] Cited:`723`

`TargetCLIP`**Image-Based CLIP-Guided Essence Transfer**<br>
_Chefer, Hila and Benaim, Sagie and Paiss, Roni and Wolf, Lior_<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2110.12427)] Cited:`33`

**CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders.**<br>
Kevin Frans, L.B. Soros, Olaf Witkowski.<br>
Arxiv 2021. [[PDF](https://arxiv.org/abs/2106.14843)] Cited:`109`

**CLIP2StyleGAN: Unsupervised Extraction of StyleGAN Edit Directions.**<br>
*Omer Kafri, Or Patashnik, Yuval Alaluf, and Daniel Cohen-Or*<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.05219)] Cited:`68`

**FEAT: Face Editing with Attention**<br>
Xianxu Hou, Linlin Shen, Or Patashnik, Daniel Cohen-Or, Hui Huang<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2202.02713)] Cited:`12`

**StyleCLIPDraw: Coupling Content and Style in Text-to-Drawing Translation**<br>
_Peter Schaldenbrand, Zhixuan Liu, Jean Oh_<br>
NeurIPS 2021 Workshop. [[PDF](https://github.com/pschaldenbrand/StyleCLIPDraw)]

**CLIPstyler: Image Style Transfer with a Single Text Condition**<br>
*Gihyun Kwon, Jong Chul Ye*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.00374)] Cited:`121`

**HairCLIP: Design Your Hair by Text and Reference Image**<br>
*Tianyi Wei, Dongdong Chen, Wenbo Zhou, Jing Liao, Zhentao Tan, Lu Yuan, Weiming Zhang, Nenghai Yu*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2112.05142)] Cited:`58`

**CLIPasso: Semantically-Aware Object Sketching**<br>
*Yael Vinker, Ehsan Pajouheshgar, Jessica Y. Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, Ariel Shamir*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2202.05822)] Cited:`51`

## Inversion-based Animation
**A good image generator is what you need for high-resolution video synthesis**<br>
Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, Sergey Tulyakov. <br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2104.15069)] Cited:`107`

**Latent Image Animator: Learning to animate image via latent space navigation.**<br>
*Yaohui Wang, Di Yang, Francois Bremond, Antitza Dantcheva.*<br>
ICLR 2022. [[PDF](https://openreview.net/forum?id=7r6kDq0mK_)]

# Image-to-Image Translation
## Supervised Image Translation

`pix2pix` **Image-to-Image Translation with Conditional Adversarial Networks**<br>
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros.<br>
CVPR 2017. [[PDF](https://arxiv.org/abs/1611.07004)] Cited:`15613`

### Semantic Image Synthesis

`CRN` **Photographic Image Synthesis with Cascaded Refinement Networks**<br>
Qifeng Chen, Vladlen Koltun. <br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1707.09405)] Cited:`866`

`pix2pixHD` **High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs**<br>
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, Bryan Catanzaro.<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.11585)] Cited:`3175`

`SPADE` **Semantic Image Synthesis with Spatially-Adaptive Normalization**<br>
Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu.<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.07291)] Cited:`1999`

`SEAN` **SEAN: Image Synthesis with Semantic Region-Adaptive Normalization**<br>
Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka.<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1911.12861)] Cited:`337`

**You Only Need Adversarial Supervision for Semantic Image Synthesis**<br>
Vadim Sushko, Edgar Schönfeld, Dan Zhang, Juergen Gall, Bernt Schiele, Anna Khoreva.<br>
ICLR 2021. [[PDF](https://arxiv.org/abs/2012.04781)] Cited:`125`

**Diverse Semantic Image Synthesis via Probability Distribution Modeling**<br>
Zhentao Tan, Menglei Chai, Dongdong Chen, Jing Liao, Qi Chu, Bin Liu, Gang Hua, Nenghai Yu.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2103.06878)] Cited:`44`

**Efficient Semantic Image Synthesis via Class-Adaptive Normalization**<br>
Zhentao Tan, Dongdong Chen, Qi Chu, Menglei Chai, Jing Liao, Mingming He, Lu Yuan, Gang Hua, Nenghai Yu.<br>
TPAMI 2021. [[PDF](https://arxiv.org/pdf/2012.04644.pdf)]

**Spatially-adaptive pixelwise networks for fast image translation.**<br>
Tamar Rott Shaham, Michael Gharbi, Richard Zhang, Eli Shechtman, Tomer Michaeli<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2012.02992)] Cited:`53`

**High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network**<br>
Jie Liang, Hui Zeng, Lei Zhang. <br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2105.09188)] Cited:`50`

### Image Inpainting
**Context encoders: Feature learning by inpainting.**<br>
_Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros_<br>
CVPR 2016. [[PDF](https://arxiv.org/abs/1604.07379)] Cited:`4506`

**Globally and locally consistent image completion.**<br>
_Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa_<br>
SIGGRAPH 2017. [[PDF](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)]

**Semantic image inpainting with deep generative models.**<br>
_Raymond A. Yeh, Chen Chen, Teck Yian Lim, Alexander G. Schwing, Mark Hasegawa-Johnson, Minh N. Do_<r>
CVPR 2017. [[PDF](https://arxiv.org/abs/1607.07539)] Cited:`1083`

**High-resolution image inpainting using multiscale neural patch synthesis**<br>
_Chao Yang, Xin Lu, Zhe Lin, Eli Shechtman, Oliver Wang, Hao Li_<br>
CVPR 2017. [[PDF](https://arxiv.org/abs/1611.09969)] Cited:`698`

**Spg-net: Segmentation prediction and guidance network for image inpainting.**<br>
_Yuhang Song, Chao Yang, Yeji Shen, Peng Wang, Qin Huang, C.-C. Jay Kuo_<br>
BMVC 2018. [[PDF](https://arxiv.org/abs/1805.03356)] Cited:`145`

**Generative image inpainting with contextual attention**<br>
_Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, Thomas S. Huang_<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1801.07892)] Cited:`1828`

**Free-form image inpainting with gated convolution.**<br>
_Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, Thomas Huang_<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1806.03589)] Cited:`1264`

**Edgeconnect: Generative image inpainting with adversarial edge learning.**<br>
_Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, Mehran Ebrahimi_<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1901.00212)] Cited:`557`

**Pluralistic Image Completion**<br>
_Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai_<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.04227)] Cited:`355`

**Rethinking image inpainting via a mutual encoder-decoder with feature equalizations.**<br>
_Hongyu Liu, Bin Jiang, Yibing Song, Wei Huang, Chao Yang_<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2007.06929)] Cited:`189`

**High-Fidelity Pluralistic Image Completion with Transformers**<br>
_Ziyu Wan, Jingbo Zhang, Dongdong Chen, Jing Liao_<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2103.14031)] Cited:`128`

**Reduce Information Loss in Transformers for Pluralistic Image Inpainting**<br>
_Qiankun Liu, Zhentao Tan, Dongdong Chen, Qi Chu, Xiyang Dai, Yinpeng Chen, Mengchen Liu, Lu Yuan, Nenghai Yu_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2205.05076)] Cited:`34`

### Attribute Editing
**Deep Identity-Aware Transfer of Facial Attributes**<br>
*Mu Li, Wangmeng Zuo, David Zhang*<br>
arxiv 2016. [[PDF](https://arxiv.org/abs/1610.05586)] Cited:`141`

### Others

### Various Applications
**Sketch Your Own GAN**<br>
Sheng-Yu Wang, David Bau, Jun-Yan Zhu<br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2108.02774)] Cited:`48`

### Super-resolution

### Example based image translation

## Unsupervised Image Transaltion
### Swapping Based
**High-Resolution Daytime Translation Without Domain Labels**<br>
I. Anokhin, P. Solovev, D. Korzhenkov, A. Kharlamov, T. Khakhulin, A. Silvestrov, S. Nikolenko, V. Lempitsky, and G. Sterkin.<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/2003.08791)] Cited:`56`

**Information Bottleneck Disentanglement for Identity Swapping**<br>
Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, Ran He<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_Information_Bottleneck_Disentanglement_for_Identity_Swapping_CVPR_2021_paper.html)]

**Swapping Autoencoder for Deep Image Manipulation**<br>
Taesung Park, Jun-Yan Zhu, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei A. Efros, Richard Zhang<br>
NeurIPS 2020. [[PDF](https://arxiv.org/abs/2007.00653)] Cited:`247`

**L2M-GAN: Learning to Manipulate Latent Space Semantics for Facial Attribute Editing**<br>
Guoxing Yang, Nanyi Fei, Mingyu Ding, Guangzhen Liu, Zhiwu Lu, Tao Xiang<br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Yang_L2M-GAN_Learning_To_Manipulate_Latent_Space_Semantics_for_Facial_Attribute_CVPR_2021_paper.html)]


### Cycle-Consistency Based
**Coupled Generative Adversarial Networks**<br>
Ming-Yu Liu, Oncel Tuzel.<br>
NeurIPS 2016 [[PDF](http://arxiv.org/abs/1606.07536)]

`UNIT` **Unsupervised Image-to-Image Translation Networks.**<br>
Ming-Yu Liu,Thomas Breuel,Jan Kautz<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1703.00848)] Cited:`2299`

`CycleGAN` **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**<br>
*Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.*<br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1703.10593)] Cited:`4339`

`DiscoGAN` **Learning to Discover Cross-Domain Relations with Generative Adversarial Networks**<br>
Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim.<br>
ICML 2017. [[PDF](https://arxiv.org/abs/1703.05192)] Cited:`1752`

`DualGAN` **DualGAN: Unsupervised Dual Learning for Image-to-Image Translation**<br>
*Zili Yi, Hao Zhang, Ping Tan, Minglun Gong.*<br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1704.02510)] Cited:`1724`

`BicycleGAN` **Toward Multimodal Image-to-Image Translation**<br>
Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, Oliver Wang, Eli Shechtman.<br>
NeurIPS 2017. [[PDF](https://arxiv.org/abs/1711.11586)] Cited:`1158`

`MUNIT` **Multimodal Unsupervised Image-to-Image Translation**<br>
Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1804.04732)] Cited:`2073`

`DRIT` **Diverse Image-to-Image Translation via Disentangled Representations**<br>
Hsin-Ying Lee, Hung-Yu Tseng, Jia-Bin Huang, Maneesh Kumar Singh, Ming-Hsuan Yang.<br>
ECCV 2018. [[PDF](https://arxiv.org/abs/1808.00948)] Cited:`270`

**Augmented cyclegan: Learning many-to-many mappings from unpaired data.**
*Amjad Almahairi, Sai Rajeswar, Alessandro Sordoni, Philip Bachman, Aaron Courville.*<br>
ICML 2018. [[PDF](https://arxiv.org/abs/1802.10151)] Cited:`367`

**MISO: Mutual Information Loss with Stochastic Style Representations for Multimodal Image-to-Image Translation.**<br>
*Sanghyeon Na, Seungjoo Yoo, Jaegul Choo.*<br>
BMVC 2020. [[PDF](https://arxiv.org/abs/1902.03938)] Cited:`12`

`MSGAN` **Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis**<br>
Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, Ming-Hsuan Yang.<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.05628)] Cited:`314`

`U-GAT-IT` **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Lee<br>
ICLR 2020. [[PDF](https://arxiv.org/abs/1907.10830)] Cited:`394`

`UVC-GAN` **UVCGAN: UNet Vision Transformer cycle-consistent GAN for unpaired image-to-image translation**<br>
*Dmitrii Torbunov, Yi Huang, Haiwang Yu, Jin Huang, Shinjae Yoo, Meifeng Lin, Brett Viren, Yihui Ren*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.02557)] Cited:`22`

### Beyond Cycle-consistency
`DistanceGAN` **One-Sided Unsupervised Domain Mapping**<br>
*Sagie Benaim, Lior Wolf*<br>
NIPS 2017. [[PDF](https://arxiv.org/abs/1706.00826)] Cited:`250`

`Council-GAN` **Breaking the Cycle - Colleagues are all you need**<br>
*Ori Nizan , Ayellet Tal*<br>
CVPR 2020. [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Nizan_Breaking_the_Cycle_-_Colleagues_Are_All_You_Need_CVPR_2020_paper.pdf)]

`ACL-GAN` **Unpaired Image-to-Image Translation using Adversarial Consistency Loss**<br>
Yihao Zhao, Ruihai Wu, Hao Dong.<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2003.04858)] Cited:`77`

`CUT` **Contrastive Learning for Unpaired Image-to-Image Translation**<br>
Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu.<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2007.15651)] Cited:`710`

**The spatially-correlative loss for various image translation tasks**<br>
_Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai._ <br>
CVPR 2021. [[PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_The_Spatially-Correlative_Loss_for_Various_Image_Translation_Tasks_CVPR_2021_paper.html)]

**Unsupervised Image-to-Image Translation with Generative Prior**<br>
Shuai Yang, Liming Jiang, Ziwei Liu and Chen Change Loy. <br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2204.03641)] Cited:`16`

### Multi-domain
`StarGAN` **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation**<br>
*Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo*<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.09020)] Cited:`2959`

`DRIT++` **DRIT++: Diverse Image-to-Image Translation via Disentangled Representations**<br>
Hsin-Ying Lee, Hung-Yu Tseng, Qi Mao, Jia-Bin Huang, Yu-Ding Lu, Maneesh Singh, Ming-Hsuan Yang.<br>
IJCV 2019. [[PDF](https://arxiv.org/abs/1905.01270)] Cited:`1039`

`StarGANv2` **StarGAN v2: Diverse Image Synthesis for Multiple Domains**<br>
*Yunjey Choi, Youngjung Uh, Jaejun Yoo, Jung-Woo Ha*<br>
CVPR 2020. [[PDF](https://arxiv.org/abs/1912.01865)] Cited:`1139`

**Smoothing the Disentangled Latent Style Space for Unsupervised Image-to-Image Translation**<br>
*Yahui Liu, Enver Sangineto, Yajing Chen, Linchao Bao, Haoxian Zhang, Nicu Sebe, Bruno Lepri, Wei Wang, Marco De Nadai*<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2106.09016)] Cited:`33`

**A Style-aware Discriminator for Controllable Image Translation**<br>
*Kunhee Kim, Sanghun Park, Eunyeong Jeon, Taehun Kim, Daijin Kim*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.15375)] Cited:`11`

`DualStyleGAN` **Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**<br>
Shuai Yang, Liming Jiang, Ziwei Liu and Chen Change Loy<br>
CVPR 2022. [[Pytorch](https://github.com/williamyang1991/DualStyleGAN)]

### Others
**Unsupervised Cross-Domain Image Generation**<br>
*Yaniv Taigman, Adam Polyak, Lior Wolf*<br>
ICLR 2017. [[PDF](https://arxiv.org/abs/1611.02200)] Cited:`874`

## Few-shot Image Translation
`FUNIT` **Few-shot unsupervised image-to-image translation.**<br>
*Ming-Yu Liu, Xun Huang, Arun Mallya, Tero Karras, Timo Aila, Jaakko Lehtinen, Jan Kautz*<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1905.01723)] Cited:`495`

**Coco-funit:Few-shot unsupervised image translation with a content conditioned style encoder.**<br>
*Kuniaki Saito, Kate Saenko, Ming-Yu Liu*<br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2007.07431)] Cited:`61`

## Few-shot Image Generation
**Attribute Group Editing for Reliable Few-shot Image Generation.**<br>
*Guanqi Ding, Xinzhe Han, Shuhui Wang, Shuzhe Wu, Xin Jin, Dandan Tu, Qingming Huang*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2203.08422)] Cited:`1`

## Style Transfer
`WCT` **Universal Style Transfer via Feature Transforms**<br>
*Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, Ming-Hsuan Yang.* <br>
NeruIPS 2017. [[PDF](https://arxiv.org/abs/1705.08086)] Cited:`753`

**Style transfer by relaxed optimal transport and self-similarity.**<br>
*Nicholas Kolkin, Jason Salavon, Greg Shakhnarovich.* <br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1904.12785)] Cited:`188`

**A Closed-Form Solution to Universal Style Transfer**<br>
*Ming Lu, Hao Zhao, Anbang Yao, Yurong Chen, Feng Xu, Li Zhang*<br>
ICCV 2019. [[PDF](https://arxiv.org/abs/1906.00668)] Cited:`60`

**Neural Neighbor Style Transfer**<br>
*Nicholas Kolkin, Michal Kucera, Sylvain Paris, Daniel Sykora, Eli Shechtman, Greg Shakhnarovich*<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.13215)] Cited:`12`

## Others
`GANgealing`**GAN-Supervised Dense Visual Alignment**<br>
William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Torralba, Alexei Efros, Eli Shechtman.<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.05143)] Cited:`41`

# Text-to-Image Synthesis
## End-to-end Training Based
**Generating images from captions with attention.**<br>
Elman Mansimov, Emilio Parisotto, Jimmy Lei Ba, Ruslan Salakhutdinov.<br>
ICLR 2016. [[PDF](https://arxiv.org/abs/1511.02793)] Cited:`364`

**Generative Adversarial Text to Image Synthesis**<br>
*Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, Honglak Lee>.*<br>
ICML 2016. [[PDF](https://arxiv.org/abs/1605.05396)] Cited:`2664`

`StackGAN` **StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks**<br>
*Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas.*<br>
ICCV 2017. [[PDF](https://arxiv.org/abs/1612.03242)] Cited:`2339`

`StackGAN++` **StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks**<br>
*Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas*<br>
TPAMI 2018. [[PDF](https://arxiv.org/abs/1710.10916)] Cited:`879`

**MirrorGAN: Learning Text-to-image Generation by Redescription**<Br>
_Tingting Qiao, Jing Zhang, Duanqing Xu, Dacheng Tao_<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1903.05854)] Cited:`414`

`AttnGAN` **AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks**<br>
*Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He.*<br>
CVPR 2018. [[PDF](https://arxiv.org/abs/1711.10485)] Cited:`1245`

`DM-GAN` **DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis**<br>
*Minfeng Zhu, Pingbo Pan, Wei Chen, Yi Yang*<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1904.01310)] Cited:`405`

`SD-GAN` **Semantics Disentangling for Text-to-Image Generation**<br>
*Guojun Yin, Bin Liu, Lu Sheng, Nenghai Yu, Xiaogang Wang, Jing Shao*<br>
CVPR 2019. [[PDF](https://arxiv.org/abs/1904.01480)] Cited:`147`

`DF-GAN` **A Simple and Effective Baseline for Text-to-Image Synthesis**<br>
*Ming Tao, Hao Tang, Fei Wu, Xiaoyuan Jing, Bingkun Bao, Changsheng Xu.*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2008.05865)] Cited:`83`

**Text to Image Generation with Semantic-Spatial Aware GAN**<br>
*Kai Hu, Wentong Liao, Michael Ying Yang, Bodo Rosenhahn*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2104.00567)] Cited:`40`

**TextFace: Text-to-Style Mapping based Face Generation and Manipulation**<br>
*Hou, Xianxu, Zhang Xiaokang, Li Yudong, Shen Linlin*<br>
TMM 2022. [[PDF](https://ieeexplore.ieee.org/document/9737433/)]

**FuseDream: Training-Free Text-to-Image Generationwith Improved CLIP+GAN Space Optimization**<br>
Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, Qiang Liu<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.01573)] Cited:`58`

**StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis**<br>
_Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, Timo Aila_<br>
arxiv 2023. [[PDF](https://arxiv.org/abs/2301.09515)] Cited:`62`

`GigaGAN`**Scaling up GANs for Text-to-Image Synthesis**<br>
_Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, Taesung Park_
CVPR 2023. [[PDF](https://arxiv.org/abs/2303.05511)] Cited:`84`

`DALLE` **Zero-Shot Text-to-Image Generation**<br>
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever.<br>
ICML 2021. [[PDF](https://arxiv.org/abs/2102.12092)] Cited:`2223`

`GLIDE` **GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**<br>
*Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen*<br>
arxiv 2021.  [[PDF](https://arxiv.org/pdf/2112.10741.pdf)] [[Pytorch](https://github.com/openai/glide-text2im)]

`DALLE2` **Hierarchical Text-Conditional Image Generation with CLIP Latents**<br>
*Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen*<br>
OpenAI 2022. [[PDF](https://cdn.openai.com/papers/dall-e-2.pdf)]

**L-Verse: Bidirectional Generation Between Image and Text**<br>
*Taehoon Kim, Gwangmo Song, Sihaeng Lee, Sangyun Kim, Yewon Seo, Soonyoung Lee, Seung Hwan Kim, Honglak Lee, Kyunghoon Bae*<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.11133)] Cited:`13`

**CLIP-GEN: Language-Free Training of a Text-to-Image Generator with CLIP**<br>
_Zihao Wang, Wei Liu, Qian He, Xinglong Wu, Zili Yi_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.00386)] Cited:`39`

**Muse: Text-To-Image Generation via Masked Generative Transformers**<br>
_Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, Dilip Krishnan_<br>
arxiv 2023. [[PDF](https://arxiv.org/abs/2301.00704)] Cited:`155`

## Multimodal Pretraining Based

**Pretraining is All You Need for Image-to-Image Translation**<br>
_Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, Fang Wen_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2205.12952)] Cited:`94`

**NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**<br>
_Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, Nan Duan_<br>
ECCV 2022.  [[PDF](https://arxiv.org/abs/2111.12417)] Cited:`151`

**NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis**<br>
_Chenfei Wu, Jian Liang, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan_<br>
NIPS 2022. [[PDF](https://arxiv.org/abs/2207.09814)] Cited:`31`

## Text-guided Image Editing
**SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations**<br>
_Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon_<br>
ICLR 2022. [[PDF](https://arxiv.org/abs/2108.01073)] Cited:`348`

**Blended Diffusion for Text-driven Editing of Natural Images**<br>
_Omri Avrahami, Dani Lischinski, Ohad Fried_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2111.14818)] Cited:`287`

**DiffusionCLIP: Text-guided Image Manipulation Using Diffusion Models**<br>
_Gwanghyun Kim, Taesung Kwon, Jong Chul Ye_<br>
CVPR 2022. [[PDF](https://arxiv.org/abs/2110.02711)] Cited:`223`

**Text2LIVE: text-driven layered image and video editing.**<br>
_Bar-Tal, Omer and Ofri-Amar, Dolev and Fridman, Rafail and Kasten, Yoni and Dekel, Tali_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2204.02491)] Cited:`127`

`Textual Inversion` **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
_Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, Daniel Cohen-Or_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2208.01618)] Cited:`435`

`DreamBooth` **DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation**<br>
_Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2208.12242)] Cited:`595`

**Prompt-to-Prompt Image Editing with Cross-Attention Control**<br>
_Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or_<br>
ICLR 2023. [[PDF](https://openreview.net/forum?id=_CDixzkzeyb)]

**Imagic: Text-Based Real Image Editing with Diffusion Models**<br>
_Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2210.09276)] Cited:`292`

**UniTune: Text-Driven Image Editing by Fine Tuning an Image Generation Model on a Single Image**<br>
_Dani Valevski, Matan Kalman, Yossi Matias, Yaniv Leviathan_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2210.09477)] Cited:`1`

**InstructPix2Pix: Learning to Follow Image Editing Instructions**<br>
_Tim Brooks, Aleksander Holynski, Alexei A. Efros_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2211.09800)] Cited:`282`

**Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models**<br>
_Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, Shiyu Chang_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2212.08698)] Cited:`19`

**Multi-Concept Customization of Text-to-Image Diffusion**<br>
_Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2212.04488)] Cited:`144`

**Zero-shot Image-to-Image Translation**<br>
[[Project](https://pix2pixzero.github.io/)]

**Null-text Inversion for Editing Real Images using Guided Diffusion Models**<br>
[[PDF](https://arxiv.org/abs/2211.09794)] [[Project](https://null-text-inversion.github.io/)] Cited:`173`

## Text-to-Video

**Imagen video: High definition video generation with diffusion models**<br>
_Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2210.02303)] [[Project](https://imagen.research.google/video/)] Cited:`366`

**Video diffusion models.**<br>
_Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2204.03458)] Cited:`447`

**Make-a-video: Text-to-video generation without text-video data**<br>
_Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, Yaniv Taigman_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2209.14792)] Cited:`331`

**Tune-A-Video: Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation**<br>
_Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, Mike Zheng Shou_<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2212.11565)] Cited:`107`

# Others

## Single Image Generation

`DIP` **Deep Image Prior**<br>
Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky. <br>
CVPR 2018 [[PDF](https://arxiv.org/abs/1711.10925)] [[Project](https://dmitryulyanov.github.io/deep_image_prior)] Cited:`2253`

`SinGAN` **SinGAN: Learning a Generative Model from a Single Natural Image**<br>
Tamar Rott Shaham, Tali Dekel, Tomer Michaeli. <br>
ICCV 2019 Best Paper. [[PDF](https://arxiv.org/abs/1905.01164)] [[Project](https://tamarott.github.io/SinGAN.htm)] Cited:`632`

`TuiGAN` **TuiGAN: Learning Versatile Image-to-Image Translation with Two Unpaired Images** <br>
Jianxin Lin, Yingxue Pang, Yingce Xia, Zhibo Chen, Jiebo Luo. <br>
ECCV 2020. [[PDF](https://arxiv.org/abs/2004.04634)] Cited:`45`

`DeepSIM` **Image Shape Manipulation from a Single Augmented Training Sample**<br>
Yael Vinker, Eliahu Horwitz, Nir Zabari , Yedid Hoshen. <br>
ICCV 2021. [[PDF](https://arxiv.org/abs/2007.01289)] [[Project](https://www.vision.huji.ac.il/deepsim/)] [[Pytorch](https://github.com/eliahuhorwitz/DeepSIM)] Cited:`10`

# Semi-supervised Learning with GAN
`SemanticGAN` **Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization**<br>
Daiqing Li, Junlin Yang, Karsten Kreis, Antonio Torralba, Sanja Fidler.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.05833)] Cited:`107`

`DatasetGAN` **DatasetGAN: Efﬁcient Labeled Data Factory with Minimal Human Effort**
Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, Sanja Fidler.<br>
CVPR 2021. [[PDF](https://arxiv.org/abs/2104.06490)] Cited:`195`


# Miscellaneous
`SemanticStyleGAN` **SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing**<br>
Yichun Shi, Xiao Yang, Yangyue Wan, Xiaohui Shen.<br>
arxiv 2021. [[PDF](https://arxiv.org/abs/2112.02236)] Cited:`36`

**Learning to generate line drawings that convey geometry and semantics**<br>
Caroline Chan, Fredo Durand, Phillip Isola.<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2203.12691)] Cited:`28`

**Synthesizing the preferred inputs for neurons in neural networks via deep generator networks.**<br>
Anh Nguyen, Alexey Dosovitskiy, Jason Yosinski, Thomas Brox, Jeff Clune. <br>
NIPS 2016. [[PDF](https://arxiv.org/abs/1605.09304)] Cited:`610`

**Generating Images with Perceptual Similarity Metrics based on Deep Networks.**<br>
Alexey Dosovitskiy, Thomas Brox<br>
NIPS 2016. [[PDF](https://arxiv.org/abs/1602.02644)] Cited:`1018`

**VectorFusion Text-to-SVG by Abstracting Pixel-Based Diffusion Models**<br>
Ajay Jain, Amber Xie, Pieter Abbeel<br>
arxiv 2022. [[PDF](https://arxiv.org/abs/2211.11319)] Cited:`18`
