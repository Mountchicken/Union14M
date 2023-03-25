## Pre-training Using MAE
We adopt the framework of [MAE](http://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) for pre-training. The code is heavily borrowed from [Masked Autoencoders: A PyTorch Implementation](https://github.com/facebookresearch/mae).

### 1. Install
```bash
conda create -n mae python=3.7
conda activate mae
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
- **Attention**: This repo is based on `timm==0.3.2`, for which a [fix](https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

### 2. Prepare dataset
- You need to prepare the dataset(s) in torchvision.datasets.ImageFolder format. The basic structure of the dataset is as follows:
    ```text
    |--dataset
        |--subfolder1
            |--image1.jpg
            |--image2.jpg
            |--...
        |--subfolder2
            |--image1.jpg
            |--image2.jpg
            |--...
    ```
- You can aslo use Union14M-U for pre-training, which is organized in ImageFolder format.

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
        --data_path Union14M-U/book32 Union14M-U/openvino /Union14M-U/CC
    ```
- Here the effective batch size is 256 (batch_size per gpu) * 1 (nodes) * 4 (gpus per node) = 1024. If memory or # gpus is limited, use --accum_iter to maintain the effective batch size, which is batch_size (per gpu) * nodes * 8 (gpus per node) * accum_iter.
- Here we use --norm_pix_loss as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off --norm_pix_loss.
- To train ViT-Base set --model mae_vit_base_patch4
- We also support tensorboard for visualization during pre-training. The learning rate, loss, and reconstructed images are logged every 200 iterations. 
Note that when using norm_pix_loss, the reconstructed images are not the original images, but the images after normalization. To use it: 
    ```bash
    tensorboard --logdir=output_dir
    ```
- The pre-training takes about 2 hours each epoch on 4 A6000 GPUs (48G).