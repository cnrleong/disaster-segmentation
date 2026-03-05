import cv2
import numpy as np

m = cv2.imread("data_raw/masks/hurricane-florence_00000000_post_disaster.png", 0)

print("shape:", m.shape)
print("unique values:", np.unique(m))
print("nonzero pixels:", np.count_nonzero(m))