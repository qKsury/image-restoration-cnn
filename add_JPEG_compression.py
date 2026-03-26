import torchvision.transforms as transforms
from io import BytesIO
import PIL.Image as Image


transformer_to_img = transforms.ToPILImage()
transformer_to_tensor = transforms.ToTensor()

def add_jpeg_compression (image_tensor, quality):
    buffer = BytesIO()


    image = transformer_to_img(image_tensor)
    image = image.convert('RGB')
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    image = Image.open(buffer).convert('RGB')
    image_tensor = transformer_to_tensor(image)

    return image_tensor
