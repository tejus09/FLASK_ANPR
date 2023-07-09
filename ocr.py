import cv2
from paddleocr import PaddleOCR
import re

class ImagePreprocessor:
    @staticmethod
    def preprocess_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)
        return blurred

class LicensePlateReader:
    def __init__(self, license_plate_pattern=r"\b[A-Z]{2}\s?[0-9]{1,2}(?:[A-Z/]){0,4}(?:\s?[A-Z]{1,3})?(?:\s?[0-9]{1,4})?\b"):
        self.license_plate_pattern = license_plate_pattern
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def read_license_plate(self, img):
        preprocessed_image = ImagePreprocessor.preprocess_image(img)
        result = self.ocr.ocr(preprocessed_image, cls=True)[0]
        txts = [line[1][0] for line in result]
        print(txts)
        if 'IND' in txts:
            txts.remove('IND')
        combined_text = " ".join(txts)
        combined_text = combined_text.replace(".", "").replace("'", "").replace('_', '')
        matches = re.findall(self.license_plate_pattern, combined_text)
        
        if matches:
            return [x.replace(" ", "") for x in matches]
        else:
            return None

