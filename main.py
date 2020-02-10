import numpy as np
import matplotlib.pyplot as plt
import image_load as im
import activation as ac
data = 'data/c.jpg'
image_pad = np.pad(im.load_image(data), (2,2), mode='constant', constant_values=0)
print('Image with padding:',image_pad.shape)


"""
Kernels (filters)
with size 3x3 for edge detection
"""

kernel_1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
kernel_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
kernel_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

print("Kernel 3x3 : ",kernel_1.shape,kernel_2.shape,kernel_3.shape)

# Preparing zero output arrays for filtered images


output_image1 = np.zeros(im.load_image(data).shape)
output_image2 = np.zeros(im.load_image(data).shape)
output_image3 = np.zeros(im.load_image(data).shape)


def pixels(arr):
    """

    Values that are more than 255

    """
    # Creating an empty array
    empty = np.empty(arr.shape)
    # Filling array with 255 value for all elements
    empty.fill(255)
    # has to be less than appropriate element in 'empty'
    result = np.where(arr < empty, arr, empty)
    return result

# Implementing convolution operation for Edge detection for GrayScale image

for i in range(image_pad.shape[0]-4):
    for j in range(image_pad.shape[1]-4):
        input_image = image_pad[i:i+3, j:j+3]
        output_image1[i,j] = np.sum(input_image*kernel_1)
        output_image2[i,j] = np.sum(input_image*kernel_2)
        output_image3[i, j] = np.sum(input_image * kernel_3)


output_image1 = pixels(ac.relu(output_image1))
output_image2 = pixels(ac.relu(output_image2))
output_image3 = pixels(ac.relu(output_image3))


def Conv():
    """




    """
    f, axarr = plt.subplots(2, 2, figsize=(10, 10))
    #plt.set_cmap('gray')
    f.suptitle('Kernel 3x3', fontsize=16)
    axarr[0, 0].imshow(im.original(data))
    axarr[0,0].set_ylabel('Original image')
    #
    axarr[0, 1].imshow(output_image1)

    #
    axarr[1, 0].imshow(output_image2)

    #
    axarr[1, 1].imshow(output_image3)

    plt.show()
Conv()

