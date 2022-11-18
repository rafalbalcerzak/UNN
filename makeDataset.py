import matplotlib as mpl
import matplotlib.image as mplimg
import pandas as pd
import numpy as np
import random
import cv2
from scipy import ndimage
from emnist import extract_training_samples
import os


# functions
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
    return page < 128


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

def makeDataset(txtFile, noise, bends, rotation, scale):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    final_directory = os.path.join(current_directory, f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


    # importing datasets
    print('importing characters')
    latinImages, latinLabels = extract_training_samples('bymerge')
    hiraganaImages = np.load('data/k49-train-imgs.npz')
    hiraganaImages = hiraganaImages['arr_0']
    hiraganaLabels = np.load('data/k49-train-labels.npz')
    hiraganaLabels = hiraganaLabels['arr_0']

    # maping datasets for using with asci codes
    print('mapping characters')
    map = pd.read_csv("data/mapping.txt", header=None, sep=' ')
    map = map[1].to_dict()
    latinNewLabels = []
    for i, l in enumerate(latinLabels):
        latinNewLabels.append(map[l])

    hiraganaNewLabels = []
    for i, l in enumerate(hiraganaLabels):
        try:
            hiraganaNewLabels.append(map[l])
        except KeyError:
            hiraganaNewLabels.append(47)

    # creating dictionary of letters from datasets
    myEmnist = {}
    for l in map:
        myEmnist[map[l]] = [x for i, x in enumerate(latinImages) if latinNewLabels[i] == map[l]]

    myHiragana = {}
    for l in map:
        myHiragana[map[l]] = [x for i, x in enumerate(hiraganaImages) if hiraganaNewLabels[i] == map[l]]

    # importing textfile into python, removing special signs like ' - * _
    lines = []  # main txt
    currentline = []
    print('importing txt file')
    with open(txtFile, encoding='utf8') as f:
        for line in f:
            for char in line:
                if len(currentline) < 80:
                    if char == '\n':
                        lines.append(currentline)
                        currentline = []
                    else:
                        if ord(char) not in map.values() and ord(char.upper()) not in map.values():
                            currentline.append(' ')
                        else:
                            if ord(char) not in map.values():
                                currentline.append(char.upper())
                            else:
                                currentline.append(char)
                else:
                    lines.append(currentline)
                    currentline = [char]

    # creation of pages
    print('Creating pages')
    latinPage = []
    hiraganaPage = []
    pagesNo = 0
    for currentLine, line in enumerate(lines):
        print(f'Processing line number: {currentLine}')
        if currentLine % 114 == 0:
            if currentLine != 0:
                latinPage = to_binary(latinPage)
                mplimg.imsave(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img/latin_page_{pagesNo}.png', latinPage, cmap='gray')
                hiraganaPage = to_binary(hiraganaPage)
                mplimg.imsave(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img/hiragana_page_{pagesNo}.png', hiraganaPage, cmap='gray')
                pagesNo += 1

            latinPage = np.zeros([32 * 114, 32 * 80])
            hiraganaPage = np.zeros([32 * 114, 32 * 80])

        latinLine = np.zeros([32, 32 * 80])
        hiraganaLine = np.zeros([32, 32 * 80])
        index = 0

        for letter in line:
            if ord(letter) in latinNewLabels:
                newLatinLetter = random.choice(myEmnist[ord(letter)])
                newHiraganaLetter = random.choice(myHiragana[ord(letter)])
                if random.random() < 0.3:
                    newLatinLetter = rotate(newLatinLetter, rotation)
                    newHiraganaLetter = rotate(newHiraganaLetter, rotation)
                if random.random() < 0.3:
                    newLatinLetter = stretch(newLatinLetter, scale)
                    newHiraganaLetter = stretch(newHiraganaLetter, scale)
                newLatinLetter = resize(newLatinLetter)
                newHiraganaLetter = resize(newHiraganaLetter)
                latinLine[0:32, index:index + 32] = newLatinLetter
                hiraganaLine[0:32, index:index + 32] = newHiraganaLetter
            index += 32

        printLine = (currentLine - (114 * pagesNo))

        latinPage[(printLine * 32):((printLine * 32) + 32), 0:(32 * 80)] = latinLine
        hiraganaPage[(printLine * 32):((printLine * 32) + 32), 0:(32 * 80)] = hiraganaLine

        currentLine += 1

    latinPage = to_binary(latinPage)
    mplimg.imsave(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img/latin_page_{pagesNo}.png', latinPage,
                  cmap='gray')
    hiraganaPage = to_binary(hiraganaPage)
    mplimg.imsave(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img/hiragana_page_{pagesNo}.png',
                  hiraganaPage, cmap='gray')

    print('adding noise')
    for i in range(pagesNo+1):
        page = cv2.imread(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img/latin_page_{i}.png')
        if bends > 0:
            page_curved = add_curve(image=page, prob=bends)
        else:
            page_curved = page
        if noise > 0:
            page_noised = s_p_noise(image=page_curved, prob=noise)
        else:
            page_noised = page_curved
        cv2.imwrite(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img/latin_page_noise_{i}.png', page_noised)

    for i in range(pagesNo+1):
        page = cv2.imread(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img/hiragana_page_{i}.png')
        if bends > 0:
            page_curved = add_curve(image=page, prob=bends)
        else:
            page_curved = page
        if noise > 0:
            page_noised = s_p_noise(image=page_curved, prob=noise)
        else:
            page_noised = page_curved
        cv2.imwrite(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img/hiragana_page_noise_{i}.png', page_noised)

    print('creating dataset')
    dataset = []

    for i in range(pagesNo+1):
        latin = cv2.imread(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/latin_img/latin_page_noise_{i}.png', cv2.IMREAD_GRAYSCALE)
        hiragana = cv2.imread(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/hiragana_img/hiragana_page_noise_{i}.png', cv2.IMREAD_GRAYSCALE)
        for x in range(80):
            for y in range(114):
                pair = []
                pair.append(latin[32 * y:32 * (y + 1), x * 32:(1 + x) * 32])
                pair.append(hiragana[32 * y:32 * (y + 1), x * 32:(1 + x) * 32])
                dataset.append(pair)

    dataset = np.array(dataset)
    print(dataset.shape)
    np.save(f'datasets/n{noise}_b{bends}_r{rotation}_s{scale}/dataset_n{noise}_b{bends}_r{rotation}_s{scale}.npy', dataset)

# txt file, noise, bends, rotation, scale
makeDataset('data/chapter1.txt', 0.07, 0.00001, 30, 15)