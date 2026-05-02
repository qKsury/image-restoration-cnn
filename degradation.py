import random
from enum import Enum
from add_noise import *



class ImageSize(Enum):
    XS = 0
    S = 1
    M = 2
    L = 3

# 'rgb_noise', 'luminance_noise', 'mixed_rgb_luminance_noise', 'strange_noise', 'chroma_noise'
all_noises = ['rgb_noise', 'luminance_noise', 'mixed_rgb_luminance_noise', 'strange_noise', 'chroma_noise' ]


PARAMS = {
    ImageSize.XS:{'noise':(0.05, 0.12)},
    ImageSize.S:{'noise': (0.07, 0.14)},
    ImageSize.M:{'noise':(0.08, 0.013)},
    ImageSize.L:{'noise':(0.1, 0.14)},
}
#я думал, что разделить на 4 размера хорошая идея, но как оказалось разница есть только между очень большими изображениями и остальными,
#жалко удалять, вдруг еще пригодится


def diagonal_size(image_tensor):
    _, height, width = image_tensor.shape
    diagonal = int(((height ** 2) + (width ** 2)) ** 0.5)
    return diagonal


def get_image_size_type(image_tensor):
    diag = diagonal_size(image_tensor)
    if diag < 550:
        raise ValueError(f"Image is too small: diagonal={diag}, minimum supported diagonal is 400")
    if diag < 1200:
        return ImageSize.XS
    if diag < 1650:
        return ImageSize.S
    if diag < 2800 :
        return ImageSize.M
    if diag < 4800:
        return ImageSize.L
    else:
        raise ValueError(f"Image is too big: diagonal={diag}, max supported diagonal is 4801")


def degradation(image_tensor):
    img_type = get_image_size_type(image_tensor)
    noise_type = random.choice(all_noises)

    match noise_type:
        case 'rgb_noise':
            return add_rgb_noise(image_tensor, random.uniform(*PARAMS[img_type]['noise']))
        case 'luminance_noise':
            return add_luminance_noise(image_tensor, random.uniform(*PARAMS[img_type]['noise']))
        case 'mixed_rgb_luminance_noise':
            return add_mixed_rgb_luminance_noise(image_tensor, random.uniform(*PARAMS[img_type]['noise']))
        case 'strange_noise':
            return add_strange_noise(image_tensor, random.uniform(*PARAMS[img_type]['noise']))
        case 'chroma_noise':
            return add_chroma_noise(image_tensor, random.uniform(*PARAMS[img_type]['noise']))
    return None