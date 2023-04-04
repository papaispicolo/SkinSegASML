# Self-supervised  Assisted  Active  Learning  for  Skin  Lesion  Segmentation
  
[![EMBC2022](https://img.shields.io/badge/arXiv-2205.07021-blue)](https://arxiv.org/abs/2205.07021)
[![EMBC2022](https://img.shields.io/badge/Conference-EMBC2022-green)](https://arxiv.org/abs/2205.07021)


Implementatino of `Self-supervised Assisted Active Learning for Skin Lesion Segmentation` paper. 

Inspired by following implementation, it provides simplified implementation of the key ideas.
- https://github.com/jacobzhaoziyuan/SAAL/
- https://github.com/omarkhaled99/skin_lesion_segmentation/


# The datasets 

ISIC segmentation data and its segmentation ground truth are resized to 512x512 images
Download the data from the link

```bash
wget https://www.dropbox.com/s/2nh2hfy23irtrhn/isic-512.zip\?dl\=0 -O isic-512.zip
```


# Pre-trained UNET model 

A pure Unet model

```
wget https://www.dropbox.com/s/b3s0wojrme61yag/unet-epoch-10.pt\?dl\=0 -O unet-epoch-10.pt
wget https://www.dropbox.com/s/nn3mwo4lgcao1sf/unet-epoch-50.pt\?dl\=0 -O unet-epoch-50.pt
wget https://www.dropbox.com/s/ygxp48s4drab0wa/unet-epoch-100.pt\?dl\=0 -O unet-epoch-100.pt
wget https://www.dropbox.com/s/5cri9ezb4j0wlw9/unet-epoch-train-10.pt\?dl\=0 -O unet-epoch-train-10.pt
wget https://www.dropbox.com/s/x1sppfgxagxzios/unet-epoch-train-50.pt\?dl\=0 -O unet-epoch-train-50.pt
wget https://www.dropbox.com/s/j8cpsa4czor4xat/unet-epoch-train-100.pt\?dl\=0 -O unet-epoch-train-100.pt
```
