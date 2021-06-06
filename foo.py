import numpy as np
import cv2
from data_generation import DataGeneration as dg

a = dg(8, 2, [0], [0])
inp, out = a.generate_sample()
for i in range(8):
    img = inp[i]
    cv2.imwrite('test/in_' + str(i) + '.jpg', img * 255)
for i in range(8):
    img = out[i]
    cv2.imwrite('test/out_' + str(i) + '.jpg', img * 255)
