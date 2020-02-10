import numpy as np
import matplotlib.pyplot as plt
from skimage import io,data,color
from skimage.transform import rescale, resize, downscale_local_mean
def load_image(image):
    """
    Reading image with skimage library
    io.imread(filename)

    """
    image = io.imread(image)
    image = resize(image,(512,512))
    image = np.array(image)
    image = image[:,:,0]
    print('Image shape:',image.shape)
    return image
def original(rgb):
    rgb = io.imread(rgb)
    rgb = resize(rgb, (512, 512))
    rgb = np.array(rgb)
    print('Image shape:', rgb.shape)
    return rgb
def display_show(im):
    """
    display image view
    """

    fig, ax = plt.subplots()
    ax.imshow(im,cmap='gray')
    ax.axis('off')
    plt.show()






