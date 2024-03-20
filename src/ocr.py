'''
https://nanonets.com/blog/ocr-with-tesseract/
'''
import ast
import os
import csv
import pandas as pd
import pytesseract
import cv2 
import sys
import subprocess
from PIL import Image, PpmImagePlugin

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract' # set this to your own tesseract file path
SAVE_FILE_PATH = './datasets/ocr/'
PNGS_DIR_PATH = "./datasets/sample_images/"


def extract_invoice_items() -> list:
    """
    executes logic to extract raw text with ocr and get formatted questions from api response
    """

    files = os.listdir(PNGS_DIR_PATH)
    index = 0
    while index < len(files):
        filename = files[index]
        if filename.endswith('.png'):
            img = cv2.imread(PNGS_DIR_PATH+filename)
            ocr_outputs = extract_raw_text(img)
        index += 1

    write_txt_to_file(ocr_outputs, str(index)+'.txt') # logging

    return ocr_outputs


def write_txt_to_file(txt: list, file_name: str) -> None:
    """
    saves the ocr outputs to txt file in subdirectory for further processing
    """

    txt.to_csv(f'{SAVE_FILE_PATH}{file_name}')

    return None


def extract_raw_text(image: PpmImagePlugin.PpmImageFile) -> str:
    """
    runs ocr package to retrieve raw text from png file image
    """
    txt = pytesseract.image_to_string(image).encode("utf-8")

    # Adding custom options
    custom_config = r'--oem 3 --psm 6' # checkout tesseract 4.0 for custom options
    txt = pytesseract.image_to_string(image, config=custom_config)

    # print(txt)  # debugging

    return str(txt, encoding='utf-8')


if __name__ == "__main__":
    extract_invoice_items()
