import torch
import torchvision.transforms as transforms


def add_noise(image_tensor, noise_level = 0.05):
    r_channel = image_tensor[0]
    g_channel = image_tensor[1]
    b_channel = image_tensor[2]

    brightness = (r_channel + g_channel + b_channel) / 3
    noise_lightrate = 1 - brightness
    noise_lightrate = noise_lightrate.unsqueeze(0)

    noise = torch.randn_like(image_tensor)
    strength = noise_level * (0.3 + 0.6 * noise_lightrate)
    noise_img = image_tensor + noise * strength

    noise_img = torch.clamp(noise_img, 0, 1)
    return noise_img
