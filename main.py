"""
Last edited
By: James Flinn

Date: 3/31/23


"""
import easyocr
import cv2
from pprint import pprint
from matplotlib import pyplot
import code

class Image_OCR:
    def __init__(self, model=None):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=True, 
            download_enabled=False, 
            model_storage_directory="easyocr_training/models/base"
            )
        
    def scan(self, image_path):
        image = cv2.imread(image_path)
        results = self.reader.readtext(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for result in results[:-1]:
            start = result[0][0]
            end = result[0][2]
            txt = result[1]
            image = cv2.rectangle(image, start, end, (0, 255, 0), 1)
            image = cv2.putText(
                image, txt, start, font, 0.5, (255, 0, 0), cv2.LINE_AA
                )
        pyplot.imshow(image)
        pyplot.show

        

if __name__ == "__main__":
    ocr = Image_OCR()
    ocr.scan("easyocr_training/test_images/mfd_menu.jpg")