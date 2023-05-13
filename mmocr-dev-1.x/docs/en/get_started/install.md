# Installation

## Prerequisites

- Linux | Windows | macOS
- Python 3.7
- PyTorch 1.6 or higher
- torchvision 0.7.0
- CUDA 10.1
- NCCL 2
- GCC 5.4.0 or higher

## Environment Setup

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation-steps). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

````{tabs}

```{code-tab} shell GPU Platform
conda install pytorch torchvision -c pytorch
```

```{code-tab} shell CPU Platform
conda install pytorch torchvision cpuonly -c pytorch
```

````

## Installation Steps

We recommend that users follow our best practices to install MMOCR. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices

**Step 0.** Install  [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
```

**Step 1.** Install MMOCR.

If you wish to run and develop MMOCR directly, install it from **source** (recommended).

If you use MMOCR as a dependency or third-party package, install it via **MIM**.

`````{tabs}

````{group-tab} Install from Source

```shell

git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -v -e .
# "-v" increases pip's verbosity.
# "-e" means installing the project in editable mode,
# That is, any local modifications on the code will take effect immediately.

```

````

````{group-tab} Install via MIM

```shell

mim install mmocr

```

````

`````

**Step 2. (Optional)** If you wish to use any transform involving `albumentations` (For example, `Albu` in ABINet's pipeline), or any dependency for building documentation or running unit tests, please install the dependency using the following command:

`````{tabs}

````{group-tab} Install from Source

```shell
# install albu
pip install -r requirements/albu.txt
# install the dependencies for building documentation and running unit tests
pip install -r requirements.txt
```

````

````{group-tab} Install via MIM

```shell
pip install albumentations>=1.1.0 --no-binary qudida,albumentations
```

````

`````

```{note}

We recommend checking the environment after installing `albumentations` to
ensure that `opencv-python` and `opencv-python-headless` are not installed together, otherwise it might cause unexpected issues. If that's unfortunately the case, please uninstall `opencv-python-headless` to make sure MMOCR's visualization utilities can work.

Refer
to [albumentations's official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for more details.

```

### Verify the installation

You may verify the installation via this inference demo.

`````{tabs}

````{tab} Python

Run the following code in a Python interpreter:

```python
>>> from mmocr.apis import MMOCRInferencer
>>> ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
>>> ocr('demo/demo_text_ocr.jpg', show=True, print_result=True)
```
````

````{tab} Shell

If you installed MMOCR from source, you can run the following in MMOCR's root directory:

```shell
python tools/infer.py demo/demo_text_ocr.jpg --det DBNet --rec CRNN --show --print-result
```
````

`````

You should be able to see a pop-up image and the inference result printed out in the console upon successful verification.

<div align="center">
    <img src="https://user-images.githubusercontent.com/24622904/187825445-d30cbfa6-5549-4358-97fe-245f08f4ed94.jpg" height="250"/>
</div>
<br />

```bash
# Inference result
{'predictions': [{'rec_texts': ['cbanks', 'docecea', 'grouf', 'pwate', 'chobnsonsg', 'soxee', 'oeioh', 'c', 'sones', 'lbrandec', 'sretalg', '11', 'to8', 'round', 'sale', 'year',
'ally', 'sie', 'sall'], 'rec_scores': [...], 'det_polygons': [...], 'det_scores':
[...]}]}
```

```{note}
If you are running MMOCR on a server without GUI or via SSH tunnel with X11 forwarding disabled, you may not see the pop-up window.
```

## Customize Installation

### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). This requires manually specifying a find-url based on PyTorch version and its CUDA version.

For example, the following command install mmcv-full built for PyTorch 1.10.x and CUDA 11.3.

```shell
pip install `mmcv>=2.0.0rc1`  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### Install on CPU-only platforms

MMOCR can be built for CPU-only environment. In CPU mode you can train (requires MMCV version >= 1.4.4), test or inference a model.

However, some functionalities are gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- SyncBatchNorm

If you try to train/test/inference a model containing above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                          Model                          |
| :-----------------------------------------------------: | :-----------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DBNet (r50dcnv2), DBNet++ (r50dcnv2), FCENet (r50dcnv2) |
|                      SyncBatchNorm                      |                      PANet, PSENet                      |

### Using MMOCR with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmocr/blob/master/docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmocr docker/
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmocr/data mmocr
```

## Dependency on MMEngine, MMCV & MMDetection

MMOCR has different version requirements on MMEngine, MMCV and MMDetection at each release to guarantee the implementation correctness. Please refer to the table below and ensure the package versions fit the requirement.

| MMOCR          | MMEngine                    | MMCV                       | MMDetection                 |
| -------------- | --------------------------- | -------------------------- | --------------------------- |
| dev-1.x        | 0.7.1 \<= mmengine \< 1.0.0 | 2.0.0rc4 \<= mmcv \< 2.1.0 | 3.0.0rc5 \<= mmdet \< 3.1.0 |
| 1.0.0          | 0.7.1 \<= mmengine \< 1.0.0 | 2.0.0rc4 \<= mmcv \< 2.1.0 | 3.0.0rc5 \<= mmdet \< 3.1.0 |
| 1.0.0rc6       | 0.6.0 \<= mmengine \< 1.0.0 | 2.0.0rc4 \<= mmcv \< 2.1.0 | 3.0.0rc5 \<= mmdet \< 3.1.0 |
| 1.0.0rc\[4-5\] | 0.1.0 \<= mmengine \< 1.0.0 | 2.0.0rc1 \<= mmcv \< 2.1.0 | 3.0.0rc0 \<= mmdet \< 3.1.0 |
| 1.0.0rc\[0-3\] | 0.0.0 \<= mmengine \< 0.2.0 | 2.0.0rc1 \<= mmcv \< 2.1.0 | 3.0.0rc0 \<= mmdet \< 3.1.0 |
