import random

from degradation import *


CROP_SIZE = 256

def image_checker(image_tensor):
    _, height, width = image_tensor.shape

    if height >= CROP_SIZE and width >= CROP_SIZE:
        return True
    return False

def get_coord_crop(image_tensor):
    _, height, width = image_tensor.shape
    left_top_height = random.randint(0, height - CROP_SIZE)
    left_top_width = random.randint(0, width - CROP_SIZE)

    right_bot_height = left_top_height + CROP_SIZE
    right_bot_width = left_top_width + CROP_SIZE

    return left_top_height, left_top_width, right_bot_height, right_bot_width


def apply_crop(image_tensor, left_top_height, left_top_width, right_bot_height, right_bot_width):
    crop = image_tensor[:, left_top_height:right_bot_height, left_top_width:right_bot_width]
    return crop