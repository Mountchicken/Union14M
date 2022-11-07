# Construct visualization for tensorboard
import numpy as np
import torch
import torchvision


def unpatchify(x):
    """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
    """
    p = 4
    h = 32 // 4
    w = 128 // 4
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


def visualize_tensor(img: torch.Tensor, reconstruction: torch.Tensor,
                     mask: torch.Tensor):
    """Construct visualization tensor for tensorboard

    Args:
        img (torch.Tensor): Normalized img tensor in shape (N, C, H, W)
        reconstruction (torch.Tensor): Normalized reconstructed image tensor in
            shape (N, L, h // patch_size * w // patch_size * 3)
        mask (torch.Tensor): Mask tensor in shape (N, L)
    Returns:
        torch.Tensor: Visualization tensor in shape (N, 3, H, W)
    """
    imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # randon select 1 images
    idx = np.random.choice(img.shape[0], 10, replace=False)
    img = img[idx].detach().cpu()
    reconstruction = reconstruction[idx].detach().cpu()
    mask = mask[idx].detach().cpu()
    # img
    img = img.mul_(imagenet_std).add_(imagenet_mean)
    # reconstruction
    reconstruction = unpatchify(reconstruction)
    reconstruction = reconstruction.mul_(imagenet_std).add_(imagenet_mean)
    # mask
    # repeat mask form (N, L) to (N, L, 192)
    mask = mask.unsqueeze(-1).repeat(1, 1, 48)

    mask = unpatchify(mask)
    # masked image
    masked_img = img * (1 - mask)
    masked_img = masked_img.mul_(imagenet_std).add_(imagenet_mean)
    tensor = torch.cat([img, masked_img, reconstruction], dim=2)
    return torchvision.utils.make_grid(tensor, nrow=5, padding=2)


if __name__ == '__main__':
    img = torch.rand(10, 3, 32, 128)
    reconstruction = torch.rand(10, 64, 192)
    mask = torch.rand(10, 64)
    vis = visualize_tensor(img, reconstruction, mask)
