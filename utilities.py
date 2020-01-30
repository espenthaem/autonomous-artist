import numpy as np
from PIL import Image


def smallest_divisor(integer, max_divisor=10):
    """
    Determines the smallest integer divisor of the input. Only checks up to a certain value.
    If no matches are found, 1 is returned.
    :param integer: int
    :return smallest divisor: int
    """
    i = 2
    while i < max_divisor + 1:
        if integer % i == 0:
            return i
        else:
            i += 1
    return 1


def alpha_blend(blend_0, blend_1, frames=10):
    """
    Performs simple alpha blending of 2 images, and returns every step of the process a PIL.Image frame. Input can be
    supplied as valid numpy array or PIL.Image object.
    :param blend_0: PIL.Image or numpy.ndarray
    :param blend_1: PIL.Image or numpy.ndarray
    :param frames: int
    :return: [PIL.Image]
    """
    if type(blend_0).__module__ == np.__name__:
        blend_0 = Image.fromarray(blend_0)

    if type(blend_1).__module__ == np.__name__:
        blend_1 = Image.fromarray(blend_0)

    output_frames = []
    for alpha in np.linspace(0, 1, int(frames)):
        output_frames.append(Image.blend(blend_0, blend_1, alpha))

    return output_frames

