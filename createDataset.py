import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


dataset =[]

for i in range(4):
    latin = cv2.imread(f'data/latin/latinNoise{i}.png', cv2.IMREAD_GRAYSCALE)
    hiragana = cv2.imread(f'data/hiragana/hiraganaNoise{i}.png', cv2.IMREAD_GRAYSCALE)
    for x in range(80):
        for y in range(114):
            pair = []
            pair.append(latin[32 * y:32 * (y + 1), x * 32:(1 + x) * 32])
            pair.append(hiragana[32 * y:32 * (y + 1), x * 32:(1 + x) * 32])
            dataset.append(pair)

dataset = np.array(dataset)
print(dataset.shape)
np.save('data/ourDataset_noise0.07_lines.npy', dataset)
