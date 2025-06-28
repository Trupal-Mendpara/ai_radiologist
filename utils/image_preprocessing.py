import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(uploaded_file, target_size=(256, 256)):

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    return image_array