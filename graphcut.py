import sys
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import PIL
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageDraw


def graphct(img):
    THRESH_MIN, THRESH_MAX = (100, 255)
    THRESH_MODE = cv2.THRESH_BINARY_INV

    FRAME = 178
    CENTERING = True
    PADDING = 5

    BG_COLOR = (255, 255, 255)
    BORDER_COLOR = (255, 255, 255)
    BORDER_WIDTH = 0

    AA = (1001, 1001)
    CONTRAST = 1.1
    SHARPNESS = 1.5
    BRIGHTNESS = 1.1
    SATURATION = 1.0
    GAMMA = 1.0

    img_src = img
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray, THRESH_MIN, THRESH_MAX, THRESH_MODE)[1]
    img_mask = cv2.merge((img_bin, img_bin, img_bin))
    cv2.imwrite("mask.jpg", img_mask)
    print(img_mask.shape)

    mask_rows, mask_cols, mask_channel = img_mask.shape
    min_x = mask_cols
    min_y = mask_rows
    max_x = 0
    max_y = 0

    for y in range(mask_rows):
        for x in range(mask_cols):
            if all(img_mask[y, x] == 255):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    rect_x = min_x
    rect_y = min_y
    rect_w = max_x - min_x
    rect_h = max_y - min_y

    mask = np.zeros(img_src.shape[:2],np.uint8)

    bg_model = np.zeros((1,65),np.float64)
    fg_model = np.zeros((1,65),np.float64)

    rect = (rect_x, rect_y, rect_w, rect_h)
    print(rect)
    cv2.grabCut(img_src, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    print(img_src.shape)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask[mask==2] = 255
    mask[mask==0] = 0
    cv2.imwrite("mask2.jpg", mask)

    img_grab = img_src * mask2[:, :, np.newaxis]
    cv2.imwrite("cut.jpg", img_grab)

    img_src_size = img_src.shape
    img_bg = np.zeros(img_src_size, dtype=np.uint8)
    img_bg[:] = BG_COLOR
    img_bg = img_bg * (1 - mask2[:, :, np.newaxis])
    img_blend = cv2.addWeighted(img_grab, 0.8, img_bg, 1, 0)

    if CENTERING:
        img_rect = img_blend[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w]
    else:
        img_rect = img_blend
        rect_w, rect_h = img_rect.shape[:2]

    rect_max = max([rect_w, rect_h])
    rect_min = min([rect_w, rect_h])

    temp_rect_max = FRAME - (PADDING * 2)
    resize_rate = temp_rect_max / rect_max
    temp_padding = int(PADDING / resize_rate)
    temp_frame_max = rect_max + (temp_padding * 2)
    img_temp = np.zeros([temp_frame_max, temp_frame_max, 3], dtype=np.uint8)
    img_temp[:] = BG_COLOR

    min_start = int((rect_max + (temp_padding * 2) - rect_min) / 2)

    if rect_w <= rect_h:
        img_temp[temp_padding: temp_padding + rect_h, min_start: min_start + rect_w] = img_rect
    else:
        img_temp[min_start: min_start + rect_h, temp_padding: temp_padding + rect_w] = img_rect

    img_aa = cv2.GaussianBlur(img_temp, AA, cv2.BORDER_TRANSPARENT)
    if GAMMA != 1.0:
        Y = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            Y[i][0] = 255 * pow(float(i) / 255, 1.0 / GAMMA)
            img_temp = cv2.LUT(img_temp, Y)

            img_aa = cv2.LUT(img_aa, Y)

    img_front = Image.fromarray(cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB)).convert('RGBA')

    img_back = Image.fromarray(cv2.cvtColor(img_aa, cv2.COLOR_BGR2RGB)).convert('RGBA')

    img_trans = Image.new('RGBA', img_front.size, (0, 0, 0, 0))

    width = img_front.size[0]
    height = img_front.size[1]

    bg_1, bg_2, bg_3 = BG_COLOR

    for x in range(width):
        for y in range(height):
            pixel = img_front.getpixel((x, y))
            if pixel[0] == bg_1 and pixel[1] == bg_2 and pixel[2] == bg_3:
                continue
            img_trans.putpixel((x, y), pixel)

    img_front = Image.new('RGBA', img_back.size, (bg_3, bg_2, bg_1, 0))
    img_front.paste(img_trans, (0, 0), img_trans)

    img_dest = Image.alpha_composite(img_back, img_front)
    img_dest = img_dest.resize((FRAME, FRAME), Image.ANTIALIAS)

    if CONTRAST != 1.0:
        img_dest = ImageEnhance.Contrast(img_dest)
        img_dest = img_dest.enhance(CONTRAST)

    if SHARPNESS != 1.0:
        img_dest = ImageEnhance.Sharpness(img_dest)
        img_dest = img_dest.enhance(SHARPNESS)

    if BRIGHTNESS != 1.0:
        img_dest = ImageEnhance.Brightness(img_dest)
        img_dest = img_dest.enhance(BRIGHTNESS)

    if SATURATION != 1.0:
        img_dest = ImageEnhance.Color(img_dest)
        img_dest = img_dest.enhance(SATURATION)

    if BORDER_WIDTH:
        border_half = BORDER_WIDTH / 2
        floor = math.floor(border_half)
        ceil = FRAME - math.ceil(border_half)
        draw = ImageDraw.Draw(img_dest)
        draw.line((0, floor)+(FRAME, floor), fill=BORDER_COLOR, width=BORDER_WIDTH)
        draw.line((ceil, 0)+(ceil, FRAME), fill=BORDER_COLOR, width=BORDER_WIDTH)
        draw.line((FRAME, ceil)+(0, ceil), fill=BORDER_COLOR, width=BORDER_WIDTH)
        draw.line((floor, FRAME)+(floor, 0), fill=BORDER_COLOR, width=BORDER_WIDTH)

    return img_dest
