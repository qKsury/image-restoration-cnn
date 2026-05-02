import os
import torch
from PIL import Image
from torchvision import transforms
from train import RestorationCNN

to_tensor = transforms.ToTensor()
to_img = transforms.ToPILImage()

robot = RestorationCNN()
robot.load_state_dict(torch.load("robot_weights_new_era_end.pth"))
robot.eval()


with torch.no_grad():
    os.makedirs('test_input', exist_ok=True)
    os.makedirs('test_output', exist_ok=True)
    for file in os.listdir('test_input'):
        img = Image.open(f'test_input/{file}').convert('RGB')
        img_tens = to_tensor(img)
        img_tens = img_tens.unsqueeze(0)
        fix_ten = robot(img_tens)
        fix_ten = fix_ten.squeeze(0)
        fix_img = to_img(fix_ten)
        fix_img.save(f'test_output/{file}')
        diff = torch.mean(torch.abs(fix_ten - img_tens.squeeze(0)))
        print(file, diff.item())