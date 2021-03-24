"""
Get datetime from images
"""

from tqdm import tqdm
from pathlib import Path
from tools import DatetimeOCR
from PIL import Image
import pytesseract as tess
PATH = Path("./images-from-video")
list_img = [str(img) for img in PATH.rglob("*.jpg")]

dt_ocr = DatetimeOCR()

for j, path_img in enumerate(list_img[:10]):
    string_dt = dt_ocr(path_img)
    print(j, string_dt)

# img = Image.open(list_img[0])
# text = tess.image_to_string(img)
# print(text)