# !/usr/bin/python
# -*- coding: utf8 -*-
# author : Suraj Chauhan

import cv2
import numpy as np


def remove_border(img: np.array, extra_crop: int = 5, CThres: int = 50, BThres: float = 60,
                  log: bool = False) -> np.array:
    """
    :param img: opencv image array
    :param extra_crop: extra cropping of pixel for fine-tuning : default=5
    :param CThres: Threshold for binary conversion of image (0-255) : default=50
    :param BThres: Threshold for border area in percentage (0-100) :default=60
    :param log: To show the log (True or False) : default=False
    :return: opencv image array
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, CThres, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_cnt = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        area_cnt.append({'area': area, 'cnt': contours[i]})
    sorted_area = sorted(area_cnt, key=lambda x: x['area'], reverse=True)
    x, y, w, h = cv2.boundingRect(sorted_area[1]['cnt'])
    per = (sorted_area[1]["area"] / sorted_area[0]["area"]) * 100
    perimg = (sorted_area[1]["area"] / (img.shape[0] * img.shape[1])) * 100
    if log:
        print(f'largest area {sorted_area[0]["area"]} second largest area {sorted_area[1]["area"]} percentage is \
            {per} perwrt to image {perimg}')

    # image = cv2.drawContours(img, [sorted_area[0]['cnt'], sorted_area[1]['cnt']], -1, (0, 255, 0), 2)
    if per > BThres and perimg > BThres:
        crop = img[y + extra_crop:y - extra_crop + h, x + extra_crop:x - extra_crop + w]
    else:
        crop = img

    return crop
