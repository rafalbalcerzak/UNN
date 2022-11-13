import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image


def s_p_noise(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add salt and pepper noise to image

    Parameters
    ----------
    image: np.ndarray
    prob: float, chance of noise
    Returns
    -------
    output: np.ndarray, noised image

    """
    output = image.copy()
    salt = 255
    pepper = 0
    print(np.int32(image.size * prob))
    for i in range(np.int32(image.size * prob)):
        y = np.random.randint(0, image.shape[0])
        x = np.random.randint(0, image.shape[1])
        if i % 2:
            output[y, x] = pepper
        else:
            output[y, x] = salt
    return output


def _generate_line(img):
    """
    Generate line

    Parameters
    ----------
    img: np.ndarray

    """
    y = np.int64(random.randint(0, img.shape[0]))
    x = np.int64(random.randint(0, img.shape[1]))
    length = np.int32(random.randint(50, 500))
    if random.random() > 0.5:
        cv2.line(img, (y, x), (y + length, x + length), color=(100), thickness=1)
    else:
        cv2.line(img, (y + length, x), (y, x + length), color=(50), thickness=1)


def add_curve(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add curves to image

    Parameters
    ----------
    image: np.ndarray
    prob: float, chance of occur line
    Returns
    -------
    output: np.ndarray, image with lines

    """
    output = image.copy()
    for i in range(np.int32(image.size * prob)):
        _generate_line(output)

    return output

for i in range(4):
    page = cv2.imread(f'data/latin/latin{i}.png')
    page_curved = add_curve(image=page, prob=0.00001)
    page_noised = s_p_noise(image=page_curved, prob=0.07)
    cv2.imwrite(f'data/latin/latinNoise{i}.png', page_noised)

for i in range(4):
    page = cv2.imread(f'data/hiragana/hiragana{i}.png')
    page_curved = add_curve(image=page, prob=0.00001)
    page_noised = s_p_noise(image=page_curved, prob=0.07)
    cv2.imwrite(f'data/hiragana/hiraganaNoise{i}.png', page_noised)