"""
Last edited
By: James Flinn

Date: 3/31/23


"""
import easyocr
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import code

class Image_OCR:
    def __init__(self, model=None):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False, 
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
            image = cv2.rectangle(image, start, end, (0, 0, 255), 1)
            image = cv2.putText(
                image, txt, start, font, 1, (255, 0 , 0), 1,
                )
        plt.imshow(image)
        plt.show()

        

if __name__ == "__main__":
    ocr = Image_OCR()
    ocr.scan("easyocr_training/test_images/Screenshot438.jpg")