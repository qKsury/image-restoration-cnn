import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

def add_gs_blur(image_tensor, kernel_size, sigma = 0.13):
    blur = transforms.GaussianBlur(kernel_size, sigma)
    blur_img = blur(image_tensor)
    return  blur_img



def add_defocus_blur(image_tensor, scale_factor):
    _, height, width = image_tensor.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    batch_img = image_tensor.unsqueeze(0)
    low_img = f.interpolate(batch_img, (new_height, new_width), mode="bilinear")
    done_img = f.interpolate(low_img, (height, width), mode="bilinear")
    done_img = done_img.squeeze(0)
    return done_img
