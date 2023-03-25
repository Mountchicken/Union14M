## Fine-tuning
We build MAERec with [MMOCR-0.x](https://github.com/open-mmlab/mmocr/tree/main). Since MMOCR-0.x is no longer matained, we will update the code to [MMOCR-1.x](https://github.com/open-mmlab/mmocr/tree/dev-1.x) ASAP.

### 1. Install
```bash
conda create -n mmocr0.x python=3.8 -y
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full
pip install mmdet
cd mmocr-0.x
pip install -r requirements.txt
pip install -v -e .
pip install timm==0.3.2
```

- **Attention**: This repo is based on `timm==0.3.2`, for which a [fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### 2. Prepare data
- In MMOCR-0.x, there are multiple [dataset formats](https://mmocr.readthedocs.io/en/dev-1.x/migration/dataset.html#text-recognition). We use the `jsonl` format for Union14M-L.
- The JSON Line format uses a dictionary-like structure to represent the annotations, where the keys filename and text store the image name and word label, respectively.
    ```json
    {"filename": "img1.jpg", "text": "OpenMMLab"}
    {"filename": "img2.jpg", "text": "MMOCR"}
    ```
- Download [Union14M-L]() 

### 3. Fine-tuning MAERec on Union14M-L
- 1. Download the [pre-trained ViT]()
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

### 4. Evaluate MAERec on Union14M-Benchmarks
- 1. Download the [pre-trained MAERec]()
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