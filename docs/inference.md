## Fine-tuning MAERec

### 1. Installation

```bash
cd mmocr-dev-1.x
conda create -n mmocr1.0 python=3.8 -y
# PyTorch 1.6 or higher is required
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
pip install timm
pip install -r requirements/albu.txt
pip install -r requirements.txt
pip install -v -e .
```

### 2. Inference MAERec on raw images
We use inferencer in MMOCR for inference. It can be used to inference on raw images, or a list of images. And it also supports visualization.

- 1. Download the [pre-trained MAERec](../README.md#42-fine-tuning)
- 2. Run the following command to inference on raw images:

    ```bash
    cd mmocr-dev-1.x
    python tools/infer.py \
        ${Input image file or folder path.} \
        --out-dir ${Output folder path.} \
        --rec configs/textrecog/maerec/maerec_b_union14m.py \
        --rec-weights ${Path to MAERec checkpoint.} \
        --device cuda \
        --show \
        --save_pred \
        --save_vis
    ```


### 3. Combine MAERec with off-the-shelf text detection models
Let's combine MAERec with [DBNet++](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnetpp) for end-to-end text recognition.

- 1. Download pretrained DBNet++ model
    ```bash
    wget https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth -O dbnetpp.pth
    ```
- 2. Run the following command to inference on raw images:

    ```bash
    cd mmocr-dev-1.x
    python tools/infer.py \
        ${Input image file or folder path.} \
        --out-dir ${Output folder path.} \
        --rec configs/textrecog/maerec/maerec_b_union14m.py \
        --rec-weights ${Path to MAERec checkpoint.} \
        --det configs/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015.py \
        --det-weights dbnetpp.pth \
        --device cuda \
        --show \
        --save_pred \
        --save_vis
    ```
