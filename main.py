import os
from PIL import Image
from torchvision import transforms as transformer
from degradation import degradation
to_tensor = transformer.ToTensor()
to_img = transformer.ToPILImage()
from make_data_set import get_coord_crop
from make_data_set import apply_crop


os.makedirs('clean_img', exist_ok= True)
os.makedirs('degr_img', exist_ok= True)
os.makedirs('images', exist_ok= True)

for name in os.listdir('images'):
    img = Image.open(f'images/{name}').convert('RGB')
    img_tensor = to_tensor(img)

    coord = get_coord_crop(img_tensor)

    deg_tensor = degradation(img_tensor)

    orig_crop = apply_crop(img_tensor, coord[0], coord[1], coord[2] ,coord[3])
    deg_crop = apply_crop(deg_tensor, coord[0], coord[1], coord[2] ,coord[3])

    orig_img = to_img(orig_crop)
    deg_img = to_img(deg_crop)


    orig_img.save(f'clean_img/{name}')
    deg_img.save(f'degr_img/{name}')