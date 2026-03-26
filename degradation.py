import random
from enum import Enum
from add_noise import add_noise
from add_blur import add_gs_blur, add_defocus_blur
from add_JPEG_compression import add_jpeg_compression




class ImageSize(Enum):
    XS = 0
    S = 1
    M = 2
    L = 3
#тип данных с ограниченным числом значений. для улучшения читаемости


class Script(Enum):
    JPEG = 0
    Blur__JPEG = 1
    JPEG__Blur = 2
    Noise__JPEG = 3
    Blur__Noise__JPEG = 4




SCENARIOS = {
    Script.JPEG: ['jpeg'],
    Script.Blur__JPEG: ['defocus_blur', 'jpeg'],
    Script.JPEG__Blur: ['jpeg', 'defocus_blur'],
    Script.Noise__JPEG: ['noise', 'jpeg'],
    Script.Blur__Noise__JPEG: ['defocus_blur', 'noise', 'jpeg']
}
weights = [2, 32, 33, 3.5, 30]

PARAMS = {
    ImageSize.XS:{'noise':(0.05, 0.09), 'scale_factor': (0.22, 0.28), 'jpeg_quality': (18, 24)},
    ImageSize.S:{'noise': (0.05, 0.09), 'scale_factor': (0.22, 0.28), 'jpeg_quality': (20, 27)},
    ImageSize.M:{'noise':(0.04, 0.07), 'scale_factor': (0.22, 0.28), 'jpeg_quality': (19, 26)},
    ImageSize.L:{'noise':(0.08, 0.12), 'scale_factor': (0.09, 0.13), 'jpeg_quality': (12, 16)},
}
#M отклиброван
#L откалиброван

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
    scripts = list(Script)

    my_scenario = random.choices(scripts, weights, k = 1)[0]
#тест номер 1 сценарий jpeg
    # my_scenario = Script.JPEG
#тест номер 2 сценарий defocus + jpeg
    # my_scenario = Script.Blur__JPEG
#тест номер 3 сценарий все
    # my_scenario = Script.Blur__Noise__JPEG
    params = PARAMS[get_image_size_type(image_tensor)]

    for effect in SCENARIOS[my_scenario]:
        match effect:
            case 'jpeg':
                image_tensor = add_jpeg_compression(image_tensor, random.randint(*params['jpeg_quality']))
            case 'defocus_blur':
                image_tensor = add_defocus_blur(image_tensor, random.uniform(*params['scale_factor']))
            case 'noise':
                image_tensor = add_noise(image_tensor, random.uniform(*params['noise']))
    return image_tensor
