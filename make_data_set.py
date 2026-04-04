import random
import PIL.Image as Image
from degradation import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

transformer_to_tensor = transforms.ToTensor()

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



class ImageRestorationDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = []

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpeg', '.png', '.jpg')):
                img = Image.open(os.path.join(self.folder_path, file)).convert('RGB')
                tensor = transformer_to_tensor(img)
                if image_checker(tensor):
                    self.files.append(file)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.folder_path, file_name)


        img = Image.open(file_path).convert("RGB")
        orig_tensor = transformer_to_tensor(img)
        degr_tensor = degradation(orig_tensor)


        coord = get_coord_crop(orig_tensor)
        orig_tensor = apply_crop(orig_tensor, coord[0], coord[1], coord[2], coord[3])
        degr_tensor = apply_crop(degr_tensor, coord[0], coord[1], coord[2], coord[3])
        return degr_tensor, orig_tensor

