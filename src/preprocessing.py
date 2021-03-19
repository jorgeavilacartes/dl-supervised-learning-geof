"""
Preprocessing is designed to run on batches 

img.shape = (batch, width, height, channels)
"""
# Standard library imports
import functools
import random 
from collections import OrderedDict

# Third party imports
import numpy as np
from skimage.transform import rescale as _rescale


# Local Imports
from .pipeline import (
    Pipeline,
    register_in_pipeline,
)

@register_in_pipeline
def rescale(img: str, *, target_size: int):
    "rescale image to (target_size, target_size, channels)"

    img_shape = img.shape

    assert len(img_shape) in {2,3}, f"Shape of image must be 3 (RGB) or 2 (GRAYSCALE)"

    if len(img_shape) == 3:
        width, height,channels = img_shape
    else:
        width, height, channels = *img_shape, 1

    max_size = max([width, height])

    rescale_factor = target_size/max_size

    if len(img_shape) == 3:
        mask = np.zeros((target_size, target_size, channels))
        rescaled_channels = [_rescale(img[:,:,channel], rescale_factor) for channel in range(channels)]
        img_rescaled = np.stack(rescaled_channels, axis=2)

        x_resc, y_resc, _ = img_rescaled.shape
        x_start, y_start = (target_size - x_resc) // 2 , (target_size -y_resc) // 2
        x_end, y_end = x_start + x_resc, y_start + y_resc 

        mask[x_start:x_end,y_start:y_end, :] = img_rescaled
    else: 
        mask = np.zeros((target_size, target_size))
        img_rescaled = _rescale(img, rescale_factor)

        x_resc, y_resc = img_rescaled.shape
        x_start, y_start = (target_size - x_resc) // 2 , (target_size -y_resc) // 2
        x_end, y_end = x_start + x_resc, y_start + y_resc 

        # Get mask 
        mask[x_start:x_end,y_start:y_end] = img_rescaled
        mask = np.expand_dims(mask, axis=-1)

    return mask
