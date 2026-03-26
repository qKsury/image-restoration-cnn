import PIL.Image as Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from degradation import degradation
import os

transformer_to_tensor = transforms.ToTensor()
transformer_to_IMG = transforms.ToPILImage()

os.makedirs("output", exist_ok=True)

for file in os.listdir("images"):
    try:
        img = Image.open(os.path.join("images", file)).convert("RGB")
        tensor = transformer_to_tensor(img)
        tensor = degradation(tensor)
        imgBAD = transformer_to_IMG(tensor)
        imgBAD.save(os.path.join('output', file))
    except ValueError as e:
        print(f"file {file} was skipped! {e}")

