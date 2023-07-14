<div align=center>

# Rethinking Scene Text Recognition: A Data Perspective

</div>

<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/4544c4ff-0f30-46b2-ae04-7bd795694df4' width=600 >
</div>

<div align=center>
  <p >Union14M is a large scene text recognition (STR) dataset collected from 17 publicly available datasets, which contains 4M of labeled data (Union14M-L) and 10M of unlabeled data (Union14M-U), intended to provide a more profound analysis for the STR community</p>

<div align=center>

[![Gradio demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Gradio-ff7c00)](https://huggingface.co/spaces/Mountchicken/MAERec-Gradio)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10Fd5x7ywL_ps4m2Am7DmvvcAflWIKCTW?usp=sharing)     


</div>


</div>
<p align="center">
   <strong><a href="#1-introduction">Introduction </a></strong> •
   <strong><a href="#34-download">Download </a></strong> •
   <strong><a href="#5-maerec">MAERec</a></strong>
   
</p>

- **Official paper will be released by next monday when arViv is ready. For now, you can check the  unofficial version [here](https://github.com/union14m/union14m.github.io/blob/main/material/Revisiting_Scene_Text_Recognition__A_Data_Perspective__arXiv_.pdf)**
- **Web page**: [https://union14m.github.io/](https://union14m.github.io/)

## 1. Introduction

- Scene Text Recognition (STR) is a fundamental task in computer vision, which aims to recognize the text in natural images. STR has been developed rapidly in recent years, and recent state-of-the-arts have shown a trend of accuracy saturation on six commonly used benchmarks (IC13, IC15, SVT, IIIT5K, SVTP, CUTE80). This is a promissing result, but it also raises a question: **Are we done with STR?** Or it's just the lack of challenges in current benchmarks that cover the drawbacks of existing methods in read-world scenarios.
<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/4ec60ac0-ca4e-4233-a196-8e9e46a0c21d' width=400 >
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/2991aa45-01cc-44da-a62f-56fd16012ab2' width=400 >
</div>

- To explore the challenges that STR models still face, we consolidate a large-scale STR dataset for analysis and identified seven open challenges. Furthermore, we propose a challenge-driven benchmark to facilitate the future development of STR. Additionally, we reveal that the utilization of massive unlabeled data through self-supervised pre-training can remarkably enhance the performance of the STR model in real-world scenarios, suggesting a practical solution for STR from a data perspective. We hope this work can spark future research beyond the realm of existing data paradigms.

## 2. Contents
- [Rethinking Scene Text Recognition: A Data Perspective](#rethinking-scene-text-recognition-a-data-perspective)
  - [1. Introduction](#1-introduction)
  - [2. Contents](#2-contents)
  - [3. Union14M Dataset](#3-union14m-dataset)
    - [3.1. Union14M-L](#31-union14m-l)
    - [3.2. Union14M-U](#32-union14m-u)
    - [3.3. Union14M-Benchmark](#33-union14m-benchmark)
    - [3.4. Download](#34-download)
  - [4. STR Models trained on Union14M-L](#4-str-models-trained-on-union14m-l)
    - [4.1. Checkpoints](#41-checkpoints)
  - [5. MAERec](#5-maerec)
    - [5.1. Pre-training](#51-pre-training)
    - [5.2. Fine-tuning](#52-fine-tuning)
    - [5.3. Evaluation](#53-evaluation)
    - [5.4. Inferencing](#54-inferencing)
    - [5.5. Demo](#55-demo)
  - [6. License](#6-license)
  - [7. Acknowledgement](#7-acknowledgement)
  - [8. Citation](#8-citation)

## 3. Union14M Dataset
### 3.1. Union14M-L
- Union14M-L contains 4M images collected from 14 public available datasets. See [Source Datasets](docs/source_dataset.md) for the details of the 14 datasets. We adopt serval strategies to refine the naive concatation of the 14 datasaets, including:
  - **Cropping**: We use minimal axis-aligned bounding box to crop the images.
  - **De-duplicate**: Some datasets contains duplicate images, we remove them.
- We also categorize the images in Union14M-L into five difficulty levels using an error voting method. 
<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/79c4b5a5-4a2f-46da-aeca-3d92e9199861' width=700 >
</div>

### 3.2. Union14M-U
- The optimal solution to improve the performance of STR in real-world scenarios is to utilize more data for training. However, labeling text images is both costly and time-intensive, given that it involves annotating sequences and needs specialized language expertise. Therefore, it would be desirable to investigate the potential of utilizing unlabeled data via self-supervised learning for STR. To this end we collect
10M unlabeled images from 3 large datasets with an IoU Voting method
<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/11711617-0fb0-42e0-a6ec-1938e2a71e61' width=600 >
</div>

### 3.3. Union14M-Benchmark
- We raise seven open challenges for STR in real-world scenarios, and propose a challenge-driven benchmark to facilitate the future development.
<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/bcfdd369-decd-4b6f-903b-3021531bd119' width=600 >
</div>

### 3.4. Download

  | Datasets                               | One Drive                                                              | Baidu Yun                                                             |
  | -------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | Union14M-L & Union14M-Benchmark (12GB) | [One Drive](https://1drv.ms/u/s!AotJrudtBr-K7xAHjmr5qlHSr5Pa?e=LJRlKQ) | [Baidu Yun](https://pan.baidu.com/s/1WiXfg9YjKiO1SzBfT14mmg?pwd=anxs) |
  | Union14M-U (36.63GB)                   | [One Drive](https://1drv.ms/f/s!AotJrudtBr-K7xGbC7wFSU62-R9m?e=ywgQAx) | [Baidu Yun](https://pan.baidu.com/s/1yOUCYgjwSB8czmZyyX56PA?pwd=4c9v) |
  | 6 Common Benchmarks (17.6MB)           | [One Drive](https://1drv.ms/u/s!AotJrudtBr-K7w8FSOI48iBI-du5?e=t8jSqN) | [Baidu Yun](https://pan.baidu.com/s/1XifQS0v-0YxEXkGTfWMDWQ?pwd=35cz) |

<!-- TODO: Add Google Drive Links -->

- The Structure of Union14M will be organized as follows:

  <details close>
  <summary><strong>Structure of Union14M-L & Union14M-Benchmark</strong></summary>

    ```text
    |--Union14M-L
      |--full_images
        |--art_curve # Images collected from the 14 datasets
        |--art_scene
        |--COCOTextV2
        |--...
      |--train_annos
        |--mmocr-0.x # annotation in mmocr0.x format
          |--train_challenging.jsonl # challenging subset
          |--train_easy.jsonl # easy subset
          |--train_hard.jsonl # hard subset
          |--train_medium.jsonl # medium subset
          |--train_normal.jsonl # normal subset
          |--val_annos.jsonl # validation subset
        |--mmocr1.0.x # annotation in mmocr1.0 format
          |--...
      |--Union14M-Benchmarks
        |--artistic
          |--imgs
          |--annotation.json # annotation in mmocr1.0 format
          |--annotation.jsonl # annotation in mmocr0.x format
        |--...
    ```

  </details>

  <details close>
  <summary><strong>Structure of Union14M-U</strong></summary>

  We store images in [LMDB](https://github.com/Mountchicken/Efficient-Deep-Learning/blob/main/Efficient_DataProcessing.md#21-efficient-data-storage-methods) format, and the structure of Union14M-U will be organized as belows.
  ```text
  |--Union14M-U
    |--book32_lmdb
    |--cc_lmdb
    |--openvino_lmdb
  ```
  </details>

## 4. STR Models trained on Union14M-L
- We train serval STR models on Union14M-L using [MMOCR-1.0](https://github.com/open-mmlab/mmocr/tree/dev-1.x)

### 4.1. Checkpoints
- Evaluated on both common benchmarks and Union14M-Benchmark. Accuracy (WAICS) in $\color{grey}{grey}$ are original implementation (Trained on synthtic datasest), and accuracay in $\color{green}{green}$ are trained on Union14M-L. All the re-trained models are trained to predict **upper & lower text, symbols and space.**

  |                                          Models                                           |                                                                                                                    Checkpoint                                                                                                                     |                     IIIT5K                     |                      SVT                       |                   IC13-1015                    |                   IC15-2077                    |                      SVTP                      |                     CUTE80                     |                      Avg.                      |
  | :---------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
  |       [ASTER](mmocr-dev-1.x/configs/textrecog/aster/aster_resnet45_6e_union14m.py)        | [GoogleDrive](https://drive.google.com/file/d/1m1uIYYxaPuq2Rb_wffZM2TTdW8ul2gr-/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1vnrEgr1BaTtYs0CGU4PR8Q?pwd=urmt) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wX3kbUqY4UW0ADE?e=dZSlYg)  | $\color{grey}{93.57}$ \ $\color{green}{94.37}$ | $\color{grey}{89.49}$ \ $\color{green}{89.03}$ | $\color{grey}{92.81}$ \ $\color{green}{93.60}$ | $\color{grey}{76.65}$ \ $\color{green}{78.57}$ | $\color{grey}{80.62}$ \ $\color{green}{80.93}$ | $\color{grey}{85.07}$ \ $\color{green}{90.97}$ | $\color{grey}{86.37}$ \ $\color{green}{88.07}$ |
  |          [ABINet](mmocr-dev-1.x/configs/textrecog/abinet/abinet_10e_union14m.py)          | [GoogleDrive](https://drive.google.com/file/d/16zyF_7GgQdLlwYjeLgYBEmLI5_xBP5rn/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/14fzeLOVqt7BVo-UYjYYyAQ?pwd=vnh6) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wZK0dCBUfJSsSGv?e=di78kG)  | $\color{grey}{95.23}$ \ $\color{green}{97.30}$ | $\color{grey}{90.57}$ \ $\color{green}{96.45}$ | $\color{grey}{93.69}$ \ $\color{green}{95.52}$ | $\color{grey}{78.86}$ \ $\color{green}{85.36}$ | $\color{grey}{84.03}$ \ $\color{green}{89.77}$ | $\color{grey}{84.37}$ \ $\color{green}{94.79}$ | $\color{grey}{87.79}$ \ $\color{green}{93.20}$ |
  |     [NRTR](mmocr-dev-1.x/configs/textrecog/nrtr/nrtr_resnet31-1by8-1by4_union14m.py)      | [Google Drive](https://drive.google.com/file/d/1pNCLWra0ACdM8TJJtM0VUeM_GyQQ2hSy/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1F0I3BF0H6gK47rsxNHrN_g?pwd=wcuc) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wiXLgM7ZLuN36By?e=QxS5h6) | $\color{grey}{91.50}$ \ $\color{green}{96.73}$ | $\color{grey}{88.25}$ \ $\color{green}{93.20}$ | $\color{grey}{93.69}$ \ $\color{green}{95.57}$ | $\color{grey}{72.32}$ \ $\color{green}{80.74}$ | $\color{grey}{77.83}$ \ $\color{green}{83.57}$ | $\color{grey}{75.00}$ \ $\color{green}{92.01}$ | $\color{grey}{83.09}$ \ $\color{green}{90.30}$ |
  |        [SATRN](mmocr-dev-1.x/configs/textrecog/satrn/satrn_shallow_5e_union14m.py)        | [Google Drive](https://drive.google.com/file/d/1mwzOgr-H9KNegeel-9qhnItxPdHbrr9T/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/108JVqH9Q-on5psjVkZdBwQ?pwd=bc4d) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wu4nY1oUNMSdDrY?e=4nNsKt) | $\color{grey}{96.00}$ \ $\color{green}{97.27}$ | $\color{grey}{91.96}$ \ $\color{green}{95.36}$ | $\color{grey}{96.06}$ \ $\color{green}{96.85}$ | $\color{grey}{80.31}$ \ $\color{green}{87.14}$ | $\color{grey}{88.37}$ \ $\color{green}{90.39}$ | $\color{grey}{89.93}$ \ $\color{green}{96.18}$ | $\color{grey}{90.43}$ \ $\color{green}{93.89}$ |
  | [SAR](mmocr-dev-1.x/configs/textrecog/sar/sar_resnet31_sequential-decoder_5e_union14m.py) | [Google Drive](https://drive.google.com/file/d/18gJxJAnokBVguI5W8FFU8yVZL0BlWXbM/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/120iVZrXlJZf2Z8i_jITneQ?pwd=cpt2) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wrU3Y_kczUK-Itb?e=w2fOCM) | $\color{grey}{95.33}$ \ $\color{green}{97.07}$ | $\color{grey}{88.41}$ \ $\color{green}{93.66}$ | $\color{grey}{93.69}$ \ $\color{green}{95.76}$ | $\color{grey}{76.02}$ \ $\color{green}{82.19}$ | $\color{grey}{83.26}$ \ $\color{green}{86.98}$ | $\color{grey}{90.28}$ \ $\color{green}{92.01}$ | $\color{grey}{87.83}$ \ $\color{green}{91.27}$ |



## 5. MAERec
- MAERec is a scene text recognition model composed of a ViT backbone and a Transformer decoder in auto-regressive style. It shows an outstanding performance in scene text recognition, especially when pre-trained on the Union14M-U through MAE.

  <div align=center>
    <img src='https://github.com/open-mmlab/mmocr/assets/65173622/f0ed4487-8064-452e-8657-b12e6f90792d' width=400 >
  </div>

- Results of MAERec on six common benchmarks and Union14M-Benchmarks

  <div align=center>
    <img src='https://github.com/open-mmlab/mmocr/assets/65173622/e697ebe0-22e5-4998-ac0d-38e543c7b400' width=800 >
  </div>

- Predictions of MAERec on some challenging examples

  <div align=center>
    <img src='https://github.com/open-mmlab/mmocr/assets/65173622/465f45b4-5f4e-4b08-962d-67f094f674e2' width=800 >
  </div>


### 5.1. Pre-training 
- ViT pretrained on Union14M-U.

  | Variants | Input Size | Patch Size | Embedding | Depth | Heads | Parameters | Download                                                                                                                                                                                                                                         |
  | -------- | ---------- | ---------- | --------- | ----- | ----- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | ViT-S    | 32x128     | 4x4        | 384       | 12    | 6     | 21M        | [GoogleDrive](https://drive.google.com/file/d/1F_wK7iAYzyz-T7-4D_dNQo1ORZ2Kesyw/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1nZL5veMyWhxpk8DGj0UZMw?pwd=xecv) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wzFMuw4YFFBFN2P?e=EvnbDo) |
  | ViT-B    | 32x128     | 4x4        | 768       | 12    | 12    | 85M        | [GoogleDrive](https://drive.google.com/file/d/1OSGZ7Jt7q4Pj3M9cUQznJpRDSDjzBVuy/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/17CjAOV-1kf1__a2RBo9NUg?pwd=3rvx) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7w2_fx4HF57MGQ6L?e=1NTxee) |
- If you want to pre-train the ViT backbone on your own dataset, check [pre-training](docs/pretrain.md)


### 5.2. Fine-tuning 
- MAERec finetuned on Union14M-L

  | Variants | Acc on Common Benchmarks | Acc on Union14M-Benchmarks | Download                                                                                                                                                                                                                                         |
  | -------- | ------------------------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | MAERec-S | 95.1                     | 78.6                       | [GoogleDrive](https://drive.google.com/file/d/1dKLS_r3_ysWK155pSmkm7NBf5ALsEJYd/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1wFhLQLrn9dm77TMpdxyNAg?pwd=trg4) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wf_KqhzYK--ITmV?e=pnz9Ld) |
  | MAERec-B | 96.2                     | 85.2                       | [GoogleDrive](https://drive.google.com/file/d/13E0cmvksKwvjNuR62xZhwkg8eQJfb_Hp/view?usp=sharing) / [BaiduYun](https://pan.baidu.com/s/1EhoJ-2WqkzOQFCNg55-KcA?pwd=5yx1) / [OneDrive](https://1drv.ms/u/s!AotJrudtBr-K7wl0So4Q3SRpGtMn?e=CiIc7A) |

- If you want to fine-tune MAERec on your own dataset, check [fine-tuning](docs/finetune.md)

### 5.3. Evaluation
- If you want to evaluate MAERec on benchmarks, check [evaluation](docs/finetune.md/#3-evaluate-maerec-on-union14m-benchmarks)

### 5.4. Inferencing
- If you want to inferencing MAERec on your raw pictures, check [inferencing](docs/inference.md)

### 5.5. Demo
- We also provide a Gradio APP for MAERec, which can be used to inferencing on your own pictures. You can run it locally or play with it on [🤗HuggingFace Spaces](https://huggingface.co/spaces/Mountchicken/MAERec-Gradio).
- To run it locally, you can run the following command:
  - 1. Install gradio and download the pretrained weights
    ```bash
    pip install gradio
    wget https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth -O dbnetpp.pth
    wget https://github.com/Mountchicken/Union14M/releases/download/Checkpoint/maerec_b_union14m.pth -O maerec_b.pth
    ```
  - 4. Run the gradio app
    ```bash
    python tools/gradio_app.py \
      --rec_config mmocr-dev-1.x/configs/textrecog/maerec/maerec_b_union14m.py \
      --rec_weight ${PATH_TO_MAEREC_B} \
      --det_config mmocr-dev-1.x/configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py \
      --det_weight ${PATH_TO_DBNETPP} \
    ```

<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/829741e1-ca5c-4be7-9a25-ce14da4f9d50' width=600 >
</div>

## 6. License
- The repository is released under the [MIT license](LICENSE).

## 7. Acknowledgement
- We sincerely thank all the constructors of the 17 datasets used in Union14M, and also the developers of MMOCR, which is a powerful toolbox for OCR research.

## 8. Citation

<!-- TODO: Update Google Colab
TODO: Update Arxiv Links
TODO: Update Arxiv Link in Gradio -->