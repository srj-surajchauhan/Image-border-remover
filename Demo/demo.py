from border_remover import remove_border
import cv2

image=cv2.imread("../Asset/1.jpg")
no_border=remove_border(image)
cv2.imshow('without border',no_border)
cv2.waitKey(0)
