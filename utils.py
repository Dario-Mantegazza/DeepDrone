import inspect
import cv2
import numpy as np


def isdebugging():
    """
        Returns true if code is being debugged
    Returns:
        Returns true if code is being debugged
    """
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def jpeg2np(image, size=None):
    """
        Converts a jpeg image in a 3d numpy array of RGB pixels and resizes it to the given size (if provided).
      Args:
        image: a compressed BGR jpeg image.
        size: a tuple containing width and height, or None for no resizing.

      Returns:
        img: the raw, resized image as a 3d numpy array of RGB pixels.
    """
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)

    return img


def time_conversion_to_nano(sec, nano):
    """
        convert time from ros timestamp to nanosecond timestamp
    Args:
        sec: seconds timestamp
        nano: nanoseconds remainder timestamp

    Returns:
        sum of nanoseconds
    """
    return (sec * 1000 * 1000 * 1000) + nano


def find_nearest(array, value):
    """
        find nearest value in array
    Args:
        array: array of values
        value: reference value

    Returns:
        min index of nearest array's element to value
    """
    return (np.abs(array - value)).argmin()