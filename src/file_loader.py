# Default imports
import os
import json
from pathlib import Path
from typing import List, Union

# Third party libraries
import cv2
import numpy as np 

# Local imports


class ImageLoader:
    """Load image from path (png, jpg, jpeg)"""
    VERSION=1

    def __init__(self, format='rgb',):
        self.format=format

    def __call__(self, img_path: str, format: str = "rgb",):
        """Load image from path ('.xml' or '.json')"""
        # read image in bgr (default opencv)
        if self.image_exists(img_path) is True:
            return self.load_img(img_path)
        else:
            raise Exception("Image {} does not exists".format(img_path))

    @staticmethod
    def image_exists(img_path: str,):
        """Return True if img_path exists"""
        return os.path.exists(img_path)

    @staticmethod
    def accepted_extension(img_path: str,):
        """Return True if img_path is either a png, jpg or jpeg file"""
        return True if any([img_path.endswith(ext) for ext in ['jpg', 'png', 'jpeg']]) else False

    def load_img(self, img_path: str,):
        """Load image in rgb or gray scale"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if self.format == "rgb":    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.format == "gray":    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
