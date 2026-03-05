import cv2

mask = cv2.imread("data_raw/masks/hurricane-florence_00000000_post_disaster.png",0)

mask = mask * 255

cv2.imwrite("mask_visual.png", mask)