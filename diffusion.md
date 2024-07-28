# Image Generation

## Series of Stable Diffusion Models

`VQ-GAN` **Taming Transformers for High-Resolution Image Synthesis**  
_Patrick Esser, Robin Rombach, Björn Ommer_  
2020.12 [[Paper](https://arxiv.org/abs/2012.09841)] [[Project](https://github.com/CompVis/taming-transformers)]  
Note: The introduction of generation in the latent space of images by GAN.

`LDM` **High-Resolution Image Synthesis with Latent Diffusion Models**  
_Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer_  
2021.12 [[Paper](https://arxiv.org/abs/2112.10752)] [[Project](https://github.com/CompVis/latent-diffusion)]  
Note: The introduction of diffusion model in the latent space of images.

`SD v1.2-1.4` **[Compvis/stable-diffusion](https://github.com/CompVis/stable-diffusion)**  
2022.8 CompVis  
Note: `512x512` resolution trained on subset of LAION-5B by LDM model. (860M UNet and 123M text encoder)

`SD v1.5` **[runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)**  
2022.10 RunwayML  
Note: Fine-tuned with `512x512` from SDv1.2 with more steps.

`SDv2` **[stabilityai/stable-diffusion-2](https://github.com/Stability-AI/stablediffusion)**  
2022.11 StabilityAI Inc.  
Notes:
- [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base): Trained with `512x512` in subset of LAION-5B. UNet config same as SDv1.5, us OpenCLIP-ViT/H as the text encoder.
- [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2): Resumed training on `768x768` with of [v-predict](https://arxiv.org/abs/2202.00512) from `stable-diffusion-2-base`.  
- [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1): 2022.12. Finetuned from SDv2.0 with a less restrictive NSFW filtering of LAION-5B dataset.
- [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip):  2023.03 Finetuned version of SD2.1, modified to accept (noisy) CLIP image embedding in addition to the text prompt, and can be used to create image variations (Examples) or can be chained with text-to-image CLIP priors. (The ability to reimage as [DALLE2.0 2022.04](https://arxiv.org/abs/2204.06125)) 

`SDXL` **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**  
Stability AI, Applied Research  
2023.07 [[Project](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)] [[Paper](https://arxiv.org/abs/2307.01952)]  
Notes: `1024x1024` pretraining with larger UNet size.

`SDXL-Turbo` **Adversarial Diffusion Distillation**  
Stability AI  
2023.11 [[Project](https://huggingface.co/stabilityai/sdxl-turbo)] [[Paper](https://stability.ai/news/stability-ai-sdxl-turbo)]

`SDXL-Lightning` **Progressive Adversarial Diffusion Distillation**  
ByetaDance Inc.  
2024.02 [[Project](https://huggingface.co/ByteDance/SDXL-Lightning)] [[Paper](https://arxiv.org/abs/2402.13929)]

`SD3.0` **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis**
Stability AI  
2024.3 [[Project](https://stability.ai/news/stable-diffusion-3)] [[Paper](https://arxiv.org/abs/2403.03206)]

## Identity Preservation Image Generation

Overview of methods for this task:
- Central Challenge: The entanglement(or Tradeoff) between Diversity(or Style) and Idendity-consistency.
1. Tuning Free: Using an ID Encoder to inject reference information into diffusion process.
2. 

`PuLID` **Pure and Lightning ID Customization via Contrastive Alignment**  
ByteDance Inc.  
2024.04. [[Paper](https://arxiv.org/abs/2404.16022)] [[Project](https://github.com/ToTheBeginning/PuLID)]

<details>
<summary>Notes</summary>
1. Trained modules: The two MLP in ID Encoder and cross attention layers in UNet.
2. How the ID features inserted into UNet decode process? By cross attention process, similar the the way that text is inserted.
3. Alignment Loss:
    1. 

</details>

- Imagine yourself
- InstantID
- Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation.

## Style Preservation Image Generation

- StyleAlign
-

## General Conditional Image Generation
- ControlNet
- IPAdapter


# Video Generation
