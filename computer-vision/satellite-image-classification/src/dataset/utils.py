import torch

import torchvision.transforms as transforms


PIXEL_CHANNEL_MEANS = (0.4914, 0.4822, 0.4465)
PIXEL_CHANNEL_STDS = (0.2023, 0.1994, 0.2010)
CLASS_MAPPER = {
    0: 'cloudy',
    1: 'desert',
    2: 'green_area',
    3: 'water',
}


def denormalize_images(
    images,
    means: list[float] = PIXEL_CHANNEL_MEANS,
    stds: list[float] = PIXEL_CHANNEL_STDS,
):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def get_default_image_transformer():
    image_transformer = transforms.Compose([
        transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=PIXEL_CHANNEL_MEANS, std=PIXEL_CHANNEL_STDS, inplace=True)
    ])
    return image_transformer
