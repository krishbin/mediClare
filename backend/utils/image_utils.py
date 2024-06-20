from PIL import Image


def compress_image(image_content, size=None):
    image = Image.open(image_content)
    if size:
        image = image.resize(512, 512)
    return image
