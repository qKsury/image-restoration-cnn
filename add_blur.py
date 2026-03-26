import torch
import torch.nn.functional as f
import torchvision.transforms as transforms

def add_gs_blur(image_tensor, kernel_size, sigma = 0.13):
    blur = transforms.GaussianBlur(kernel_size, sigma)
    blur_img = blur(image_tensor)
    return  blur_img
#kernel_size нечетное число, чем больше тем сильнее искажение, радиус размытия
#sigma число от 0 до бесконечности чем больше тем сильнее искажение



def add_defocus_blur(image_tensor, scale_factor):
    _, height, width = image_tensor.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    batch_img = image_tensor.unsqueeze(0)
    low_img = f.interpolate(batch_img, (new_height, new_width), mode="bilinear")
    done_img = f.interpolate(low_img, (height, width), mode="bilinear")
    done_img =done_img.squeeze(0)
    return done_img

# scale_factor число от 0 до 1 чем меньше, тем сильнее портит
# это число на которое мы умножаем длину и ширину изображения
# чтобы уменьшить а потом увеличить его