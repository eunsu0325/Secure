"""Transform functions for COCONUT"""

import torch
from PIL import Image
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support grayscale image.')

        tensor = tensor.view(c, h*w)
        idx = tensor > 0
        t = tensor[idx]

        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s+1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


def get_scr_transforms(train=True, imside=128, channels=1):
    """Get transforms for SCR training/evaluation

    Args:
        train: If True, applies data augmentation for training
        imside: Target image size
        channels: Number of output channels (1 for grayscale, 3 for RGB)

    Returns:
        Composed transformation pipeline
    """
    if not train:
        return T.Compose([
            T.Resize(imside),
            T.ToTensor(),
            NormSingleROI(outchannels=channels)
        ])
    else:
        return T.Compose([
            T.Resize(imside),
            T.RandomChoice(transforms=[
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(size=imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice(transforms=[
                    T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,
                                   expand=False, center=(0.5*imside, 0.0)),
                    T.RandomRotation(degrees=10, interpolation=Image.BICUBIC,
                                   expand=False, center=(0.0, 0.5*imside)),
                ]),
            ]),
            T.ToTensor(),
            NormSingleROI(outchannels=channels)
        ])