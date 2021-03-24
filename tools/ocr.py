"""
Get datetime from image

https://tesseract-ocr.github.io/tessdoc/Compiling.html#linux
"""
#TODO: Decidir si usar PIL o cv2
#TODO: encontrar un preprocesamiento para extraer adecuadamente el string de la imagen
import os
import re
import time
from collections import OrderedDict
from PIL import Image
# import cv2
import pytesseract

PATTERNS = OrderedDict(
    {
        "date": r'[0-9]{4}/[0-9]{2}/[0-9]{2}',
        "time": r'[0-9]{2}:[0-9]{2}:[0-9]{2}'
    }
)
class DatetimeOCR:  

    def __init__(self,):
        # self.config_ocr = r'--oem 3 --psm 6'
        self.config_ocr = r'--psm 10'

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
            # return cv2.imread(path_img)
        else:
            raise Exception("path {!r} does not exist".format(path_img))

    def datetime_from_image(self, img):
        """Specific function for the kind of image in this problem"""
        # Crop part of the image with date
        img = self.crop_image(img)
        
        self.show_img(img)

        # Use OCR
        string = self.apply_ocr(img)

        # Get date and time
        date = self.get_date(string)
        time = self.get_time(string)

        Image.Image.close(img)
        return " ".join([date, time])

    
    def apply_ocr(self,img):
        """Apply OCR to the image"""
        img = self.thresholding(img)
        return pytesseract.image_to_string(img, config=self.config_ocr)

    @staticmethod
    def crop_image(
            img,
            left: int = 0, 
            top: int = 0, 
            right: int = 500, 
            bottom: int = 40
        ):
        """Crop image. Default values correspond to the datetime area in this problem"""
        return img.crop((left, top, right, bottom)) 

    # cv2
    # @staticmethod
    # def crop_image(
    #         img,
    #         left: int = 0, 
    #         top: int = 0, 
    #         right: int = 400, 
    #         bottom: int = 40
    #     ):
    #     """Crop image. Default values correspond to the datetime area in this problem"""
    #     return img[left:right, top:bottom,:] 

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
        
    @staticmethod
    def thresholding(img):
        threshold = 40
        return img.point(lambda p: p > threshold and 255)

    @staticmethod
    def show_img(img):
        img.show()
        time.sleep(1)
        