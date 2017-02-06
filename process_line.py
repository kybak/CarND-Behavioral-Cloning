from skimage import io
import numpy as np
from PIL import Image
from resizeimage import resizeimage


def process_line(data, batch_size):
    height = 66
    width = 200
    channels = 3

    x = np.zeros((batch_size, height, width, channels))
    y = np.zeros((batch_size, 1))

    for i in range(0, batch_size):
        if abs(float(data[i][3])) > 0.05:
            with open(data[i][0], 'r+b') as f:
                with Image.open(f) as image:
                    resized = resizeimage.resize_cover(image, [200, 66])


            x[i] = resized
            y[i] = data[i][3]

    return x, y

