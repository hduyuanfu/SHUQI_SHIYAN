import win32gui
import numpy
from PIL import ImageGrab
import cv2

block_width, block_height = 16, 16


def crop_block(hole_img, x, y):
    x1, y1 = x * block_width, y * block_height
    x2, y2 = x1 + block_width, y1 + block_height
    return hole_img.crop((x1, y1, x2, y2))
    

def pil_to_cv(img):
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def get_frame():
    
    hwnd = 0x00071150
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    rect = (left, top, right, bottom)
    img = ImageGrab.grab().crop(rect)

    blocks_x = int((right - left) / block_width)
    blocks_y = int((bottom - top) / block_height)

    blocks_img = [[0 for i in range(blocks_y)] for i in range(blocks_x)]

    for y in range(blocks_y):
        for x in range(blocks_x):
            blocks_img[x][y] = crop_block(img, x, y)

    return img, blocks_img, blocks_x, blocks_y, block_width, block_height, left, top, right, bottom

