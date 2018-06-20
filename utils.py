import inspect
import cv2
import numpy as np


def isdebugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def jpeg2np(image, size=None):
    """Converts a jpeg image in a 2d numpy array of RGB pixels and resizes it to the given size (if provided).
      Args:
        image: a compressed BGR jpeg image.
        size: a tuple containing width and height, or None for no resizing.

      Returns:
        the raw, resized image as a 2d numpy array of RGB pixels.
    """
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    # TODO eliminate conversion everywhere
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)

    return img


# method to convert time

def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


# find nearest value in array
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()