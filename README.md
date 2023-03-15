# Union14M Dataset

<div align=center>
  <img src='github/cover.png' width=600 >
</div>
<div align=center>
  <p >Union14M is a large scene text recognition (STR) dataset collected from 17 publicly available datasets, which contains 4M of labeled data (Union14M-L) and 10M of unlabeled data (Union14M-U), intended to provide a more profound analysis for the STR community</p>
</div>
<p align="center">
   <strong><a href="#sota">arXiv </a></strong> •
   <strong><a href="./papers.md">download </a></strong> •
   <strong><a href="./datasets.md">pre-training </a></strong> •
   <strong><a href="#code">fine-tuning</a></strong>
</p>

## 1. Introduction
- Scene Text Recognition (STR) is a fundamental task in computer vision, which aims to recognize the text in natural images. STR has been developed rapidly in recent years, and recent state-of-the-arts have shown a trend of accuracy saturation on six commonly used benchmarks (IC13, IC15, SVT, IIIT5K, SVTP, CUTE80). This is a promissing result, but it also raises a question: **Are we done with STR?** Or it's just the lack of challenges in current benchmarks that cover the drawbacks of existing methods in read-world scenarios.
<div align=center>
  <img src='github/acc_trend.png' width=400 >
  <img src='github/benchmark_analysis.png' width=400 >
</div>

- To explore the challenges that STR models still face, we consolidate a large-scale STR dataset for analysis and identified seven open challenges. Furthermore, we propose a challenge-driven benchmark to facilitate the future development of STR. Additionally, we reveal that the utilization of massive unlabeled data through self-supervised pre-training can remarkably enhance the performance of the STR model in real-world scenarios, suggesting a practical solution for STR from a data perspective. We hope this work can spark future research beyond the realm of existing data paradigms.

## 2. Contents
- [Union14M Dataset](#union14m-dataset)
  - [1. Introduction](#1-introduction)
  - [2. Contents](#2-contents)
  - [3. Union14M Dataset](#3-union14m-dataset)
    - [3.1. Union14M-L](#31-union14m-l)
    - [3.2. Union14M-U](#32-union14m-u)
    - [3.3. Union14M-Benchmark](#33-union14m-benchmark)
    - [3.4. Download](#34-download)

## 3. Union14M Dataset
### 3.1. Union14M-L
- Union14M-L contains 4M images collected from 14 public available datasets. See [Source Datasets](docs/source_dataset.md) for the details of the 14 datasets. We adopt serval strategies to refine the naive concatation of the 14 datasaets, including:
  - **Cropping**: We use minimal axis-aligned bounding box to crop the images.
  - **De-duplicate**: Some datasets contains duplicate images, we remove them.
- We also categorize the images in Union14M-L into five difficulty levels using an error voting method. 
<div align=center>
  <img src='github/union14m-l.png' width=700 >
</div>

### 3.2. Union14M-U
- The optimal solution to improve the performance of STR in real-world scenarios is to utilize more data for training. However, labeling text images is both costly and time-intensive, given that it involves annotating sequences and needs specialized language expertise. Therefore, it would be desirable to investigate the potential of utilizing unlabeled data via self-supervised learning for STR. To this end we collect
10M unlabeled images from 3 large datasets, using an IoU Voting method
<div align=center>
  <img src='github/union14m-u.png' width=600 >
</div>

### 3.3. Union14M-Benchmark
- We raise seven open challenges for STR in real-world scenarios, and propose a challenge-driven benchmark to facilitate the future development.
<div align=center>
  <img src='github/benchmark_image.png' width=600 >
</div>

### 3.4. Download