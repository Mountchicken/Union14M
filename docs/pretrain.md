## Pre-training Using MAE
We adopt the framework of [MAE](http://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) for pre-training. The code is heavily borrowed from [Masked Autoencoders: A PyTorch Implementation](https://github.com/facebookresearch/mae).

### 1. Installation
```bash
cd mae/
conda create -n mae python=3.8
conda activate mae
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
- **Attention**: The pre-training code is based on `timm==0.3.2`, for which a [fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. Add the below code to `timm/models/layers/helpers.py`:
    ```python
    import torch

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        from torch._six import container_abcs
    else:
        import collections.abc as container_abcs
    ```

### 2. Prepare dataset
- We support two types of datasets: ImageFolder and LMDB.
  - torchvision.datasets.ImageFolder format:
      ```text
      |--dataset
          |--book32
              |--image1.jpg
              |--image2.jpg
              |--...
          |--openvino
              |--image1.jpg
              |--image2.jpg
              |--...
      ```
  - LMDB format. To know more about LMDB structure and how to create LMDB, you should not miss this [repo](https://github.com/Mountchicken/Efficient-Deep-Learning/blob/main/Efficient_DataProcessing.md#21-efficient-data-storage-methods).
      ```text
      |--dataset
          |--book32
            |--data.mdb
            |--lock.mdb
          |--openvino
            |--data.mdb
            |--lock.mdb
          |--cc
            |--data.mdb
            |--lock.mdb
      ```

### 3. Pre-training
- Pre-training ViT-Small on Union14M-U with 4 gpus:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
        --nproc_per_node=4 main_pretrain.py \
        --batch_size 256 \
        --model mae_vit_small_patch4 \
        --mask_ratio 0.75 \
        --epochs 20 \
        --warmup_epochs 2 \
        --norm_pix_loss \
        --blr 1.5e-4 \
        --weight_decay 0.05 \
        --data_path ../data/Union14M-U/book32_lmdb ../data/Union14M-U/cc_lmdb ../data/Union14M-U/openvino_lmdb 
    ```
- To pretrain ViT-Base, use `--model mae_vit_base_patch4`.
- Here the effective batch size is 256 (batch_size per gpu) * 1 (nodes) * 4 (gpus per node) = 1024. If memory or # gpus is limited, use --accum_iter to maintain the effective batch size, which is batch_size (per gpu) * nodes * 8 (gpus per node) * accum_iter.
- Here we use --norm_pix_loss as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off --norm_pix_loss.
- To train ViT-Base set --model mae_vit_base_patch4
- We also support tensorboard for visualization during pre-training. The learning rate, loss, and reconstructed images are logged every 200 iterations. 
Note that when using norm_pix_loss, the reconstructed images are not the original images, but the images after normalization. To use it: 
    ```bash
    tensorboard --logdir=output_dir
    ```
<div align=center>
  <img src='https://github.com/open-mmlab/mmocr/assets/65173622/2cbc0f73-a1b1-441b-b000-c598138bb7e5' width=600 >
</div>

- The pre-training takes about 2 hours each epoch on 4 A6000 GPUs (48G).