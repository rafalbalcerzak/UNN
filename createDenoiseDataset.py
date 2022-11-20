from emnist import extract_training_samples
import matplotlib.pyplot as plt
import random
from scipy import ndimage
import cv2
from tqdm import tqdm
import numpy as np

def s_p_noise(image: np.ndarray, prob: float) -> np.ndarray:
    output = image.copy()
    salt = 255
    pepper = 0
    for i in range(np.int32(image.size * prob)):
        y = np.random.randint(0, image.shape[0])
        x = np.random.randint(0, image.shape[1])
        if i % 2:
            output[y, x] = pepper
        else:
            output[y, x] = salt
    return output

def rotate(img, factor):
    angle = random.uniform(-1*factor, factor)
    img = ndimage.rotate(img, angle, reshape=False)
    return img


def stretch(img, factor):
    percentage = random.uniform(-1*factor, factor)
    if random.random() < 0.5:
        img = cv2.resize(img, (0, 0), fx=1 + (percentage / 100), fy=1)
    else:
        img = cv2.resize(img, (0, 0), fx=1, fy=1 + (percentage / 100))
    return img


def resize(img):
    h, w = img.shape
    back = np.zeros([32, 32])
    yoff = round((32 - h) / 2)
    xoff = round((32 - w) / 2)
    back[yoff:yoff + h, xoff:xoff + w] = img
    return back


def to_binary(page):
    return page > 128


latinImages, latinLabels = extract_training_samples('bymerge')
rotation = 30
scale =15

print(' ')
print('Rotating and scaling:')
distortedDataset = []
for i in tqdm(range(len(latinImages))):
    if random.random() < 0.3:
        rotatedImg = rotate(latinImages[i], rotation)
    else:
        rotatedImg = latinImages[i]

    if random.random() < 0.3:
        scaledImg = stretch(rotatedImg, scale)
    else:
        scaledImg = rotatedImg

    resized = resize(scaledImg)
    distortedDataset.append(to_binary(resized))

print('Noising:')
noisedDataset = []
for i in tqdm(range(len(distortedDataset))):
    noisedDataset.append(s_p_noise(image=distortedDataset[i], prob=0.07))


plt.imshow(distortedDataset[0], cmap='binary')
plt.show()
plt.imshow(noisedDataset[0], cmap='binary')
plt.show()

np.save(f'datasets/denoiseAE/withNoise.npy', noisedDataset)
np.save(f'datasets/denoiseAE/withoutNoise.npy', distortedDataset)
