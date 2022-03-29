import numpy as np
import math
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib import pyplot as plt

def normalize_image(image):
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    return normalized

def compute_derivative_x(image):

    height, width = image.shape

    tmp_front = np.copy(image)
    tmp_back = np.copy(image)

    tmp_front = np.roll(tmp_front, -1, axis=0)
    tmp_back = np.roll(tmp_back, 1, axis=0)

    tmp_front[height-1] = image[height-1]
    tmp_back[0] = image[0]

    tmp = (tmp_front-tmp_back)

    return tmp


def compute_derivative_y(image):
    height, width = image.shape

    tmp_front = np.copy(image)
    tmp_back = np.copy(image)

    tmp_front = np.roll(tmp_front, -1, axis=1)
    tmp_back = np.roll(tmp_back, 1, axis=1)

    tmp_front[:,width - 1] = image[:,width - 1]
    tmp_back[:,0] = image[:,0]

    tmp = (tmp_front - tmp_back)

    return tmp

def get_gradient_magnitude_orientation(image):

    gradx = compute_derivative_x(image)
    grady = compute_derivative_y(image)

    magnitude_result = np.sqrt((gradx * gradx) + (grady * grady))

    grady[grady == 0.] = 1.
    gradx[gradx == 0.] = 1.
    orientation_result = ((np.arctan2(gradx , grady) * (180 / math.pi)).astype(np.int32))
    return magnitude_result, orientation_result

def compareHistogram(hist1, hist2):
    result = sum([(i-j) ** 2 for i, j in zip(hist1, hist2)])
    return result