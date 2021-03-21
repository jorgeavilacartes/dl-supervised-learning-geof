"""
Get datetime from image

https://tesseract-ocr.github.io/tessdoc/Compiling.html#linux
"""

import os
import re
from collections import OrderedDict
from PIL import Image
import pytesseract

PATTERNS = OrderedDict(
    {
        "date": r'[0-9]{4}/[0-9]{2}/[0-9]{2}',
        "time": r'[0-9]{2}:[0-9]{2}:[0-9]{2}'
    }
)
class DatetimeOCR:  

    def __call__(self, path_img):
        # Load image
        img = self.load_img(path_img)
        
        # Get datetime from image
        dt = self.datetime_from_image(img)

        return dt

    def _path_exists(self, path):
        if not os.path.exists(path):
            raise Exception("{} does not exists".format(path))
        return True

    def load_img(self, path_img):
        """Load an image as PIL object if this existst"""
        if self._path_exists(path_img):
            return Image.open(path_img)
        else:
            raise Exception("path {!r} does not exist".format(path_img))

    def datetime_from_image(self, img):
        """Specific function for the kind of image in this problem"""
        # Crop part of the image with date
        img = self.crop_image(img)

        # Use OCR
        string = self.apply_ocr(img)

        # Get date and time
        date = self.get_date(string)
        time = self.get_time(string)

        return " ".join([date, time])

    @staticmethod
    def apply_ocr(img):
        """Apply OCR to the image"""
        return pytesseract.image_to_data(img, lang="eng")

    @staticmethod
    def crop_image(
            img,
            left: int = 0, 
            top: int = 0, 
            right: int = 400, 
            bottom: int = 40
        ):
        """Crop image. Default values correspond to the datetime area in this problem"""
        return img.crop((left, top, right, bottom)) 

    @staticmethod
    def get_date(string):
        """Get date pattern YYYY/MM/DD"""
        try:
            return re.findall(PATTERNS.get("date"), string)[0]
        except: 
            return '9999/99/99'

    @staticmethod
    def get_time(string):    
        """Get time pattern HH:MM:SS"""
        try:
            return re.findall(PATTERNS.get("time"), string)[0]
        except: 
            return '99:99:99'
        