import torch
import torchvision.transforms as transforms
import random



# возможно нужно будет понизить мощность в 3 раза из-за того, что я отдельно каждый канал порчу
def add_rgb_noise(image_tensor, noise_level = 0.05):

    r_channel = image_tensor[0]
    g_channel = image_tensor[1]
    b_channel = image_tensor[2]



    r_coef = random.uniform(0.8,1.2)
    g_coef = random.uniform(0.8, 1.2)
    b_coef = random.uniform(0.9, 1.3)



    r_noise = torch.randn_like(r_channel) * r_coef
    g_noise = torch.randn_like(g_channel) * g_coef
    b_noise = torch.randn_like(b_channel) * b_coef

    brightness = (r_channel + g_channel + b_channel) / 3
    darkness = 1 - brightness
    shadow_map = 0.5 + darkness

    noise_tensor = torch.stack([
        r_channel + r_noise * noise_level * shadow_map,
        g_channel + g_noise * noise_level * shadow_map,
        b_channel + b_noise * noise_level * shadow_map
    ],dim = 0)

    noise_img = torch.clamp(noise_tensor, 0, 1)
    return noise_img






def add_luminance_noise(image_tensor, noise_level = 0.05):

    r_channel = image_tensor[0]
    g_channel = image_tensor[1]
    b_channel = image_tensor[2]


    luminance_noise = torch.randn_like(r_channel)

    brightness = (r_channel + g_channel + b_channel) / 3
    darkness = 1 - brightness
    shadow_map = (0.5 + darkness)

    noise_tensor = torch.stack([
        r_channel + luminance_noise * noise_level * shadow_map,
        g_channel + luminance_noise * noise_level * shadow_map,
        b_channel + luminance_noise * noise_level * shadow_map
    ], dim=0)

    noise_img = torch.clamp(noise_tensor, 0, 1)
    return noise_img



def add_mixed_rgb_luminance_noise(image_tensor, noise_level = 0.05):

    noise_img = add_rgb_noise(image_tensor, noise_level * 0.3)
    noise_img = add_luminance_noise(noise_img, noise_level * 0.6)
    return noise_img




def add_strange_noise(image_tensor, noise_level = 0.05):
    r_channel = image_tensor[0]
    g_channel = image_tensor[1]
    b_channel = image_tensor[2]

    r_coef = random.uniform(0.8, 1.2)
    g_coef = random.uniform(0.8, 1.2)
    b_coef = random.uniform(0.9, 1.3)

    r_noise = torch.randn_like(r_channel) * r_coef * r_channel
    g_noise = torch.randn_like(g_channel) * g_coef * g_channel
    b_noise = torch.randn_like(b_channel) * b_coef * b_channel

    brightness = (r_channel + g_channel + b_channel) / 3
    darkness = 1 - brightness
    shadow_map = 0.5 + darkness

    noise_tensor = torch.stack([
        r_channel + r_noise * noise_level * shadow_map * 0.7,
        g_channel + g_noise * noise_level * shadow_map * 0.7,
        b_channel + b_noise * noise_level * shadow_map * 0.7
    ], dim=0)

    noise_img = torch.clamp(noise_tensor, 0, 1)
    return noise_img




def add_chroma_noise(image_tensor, noise_level = 0.05):
    noise_tensor = add_rgb_noise(image_tensor,  noise_level)
    first_avg =  (image_tensor + noise_tensor) / 2
    return first_avg


























