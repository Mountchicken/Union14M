## Fine-tuning MAERec
We build MAERec with [MMOCR-1.0](https://github.com/open-mmlab/mmocr/tree/dev-1.x).

### 1. Install

```bash
conda create -n mmocr1.0 python=3.8 -y
# PyTorch 1.6 or higher is required
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
cd mmocr-dev-1.x
pip install timm==0.3.2
pip install -r requirements/albu.txt
pip install -r requirements.txt
pip install -v -e .
```

- **Attention ❗️**: This repo is based on `timm==0.3.2`, for which a [fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### 2. Fine-tuning MAERec on Union14M-L
- 1. Download the [pre-trained ViT](../README.md#41-pre-training)
- 2. Modify the config file `mmocr-0.x/configs/textrecog/maerec/maerec_b` to set the `pretrained` field in `model.backbone` to the path of the pre-trained ViT model.
- 3. Modify the config file `mmocr-0.x/configs/_base_/recog_datasets/Union14M_train.py` to set the `train_root` field to the path of the Union14M-L dataset.
- 4. Modify the config file `mmocr-0.x/configs/_base_/recog_datasets/Union14M_benchmark.py` to set the `test_root` field to the path of the Union14M-Benchmarks.
- 5. Run the following command to fine-tune MAERec on Union14M-L.
    ```bash
    cd mmocr-0.x
    # training with single GPU
    python tools/train.py configs/textrecog/maerec/maerec_b.py
    # training with multiple GPUs (8 GPUs in this example)
    bash tools/dist_train.sh configs/textrecog/maerec/maerec_b.py 8
    ```

### 3. Evaluate MAERec on Union14M-Benchmarks
- 1. Download the [pre-trained MAERec](../README.md#42-fine-tuning)
- 2. Modify the config file `mmocr-0.x/configs/_base_/recog_datasets/Union14M_benchmark.py` to set the `test_root` field to the path of the Union14M-Benchmarks.
- 3. Run the following command to evaluate MAERec on Union14M-Benchmarks.
    ```bash
    cd mmocr-0.x
    # evaluation with single GPU
    python tools/test.py configs/textrecog/maerec/maerec_b.py \
        {PATH TO PRETRAINED MAEREC} \
        --eval acc
    # evaluation with multiple GPUs (8 GPUs in this example)
    bash tools/dist_test.sh configs/textrecog/maerec/maerec_b.py \
        {PATH TO PRETRAINED MAEREC} \
        8 --eval acc
    ```