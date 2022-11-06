import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import cv2
from scipy import ndimage
from emnist import extract_training_samples

# functions
def rotate(img):
    angle =random.uniform(-30, 30)
    img = ndimage.rotate(img, angle, reshape=False )
    return img

def stretch(img):
    percentage =random.uniform(-15, 15)
    if random.random() < 0.5:
        img = cv2.resize(img, (0, 0), fx=1+(percentage/100), fy=1)
    else:
        img = cv2.resize(img, (0, 0), fx=1, fy=1+(percentage/100))
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

#importing datasets
latinImages, latinLabels = extract_training_samples('bymerge')
hiraganaImages = np.load('data/k49-train-imgs.npz')
hiraganaImages = hiraganaImages['arr_0']
hiraganaLabels = np.load('data/k49-train-labels.npz')
hiraganaLabels = hiraganaLabels['arr_0']

# maping datasets for using with asci codes
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

with open('data/chapter1.txt', encoding='utf8')as f:
    for line in f:
        for char in line:
            if len(currentline) <80:
                if char == '\n':
                    lines.append(currentline)
                    currentline = []
                else:
                    if ord(char) not in map.values() and ord(char.upper())not in map.values():
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
latinPage = []
hiraganaPage = []
pagesNo = 0
for currentLine, line in enumerate(lines):
    print(f'Processing line number: {currentLine}')
    if currentLine % 114 == 0:
        if currentLine != 0:
            latinPage =to_binary(latinPage)
            mpl.image.imsave(f'data/latin/latin{pagesNo}.png', latinPage, cmap='gray')
            hiraganaPage = to_binary(hiraganaPage)
            mpl.image.imsave(f'data/hiragana/hiragana{pagesNo}.png', hiraganaPage, cmap='gray')
            pagesNo +=1

        latinPage = np.zeros([32 * 114, 32 * 80])
        hiraganaPage = np.zeros([32 * 114, 32 * 80])

    latinLine = np.zeros([32, 32*80])
    hiraganaLine = np.zeros([32, 32 * 80])
    index = 0

    for letter in line:
        if ord(letter) in latinNewLabels:
            newLatinLetter = random.choice(myEmnist[ord(letter)])
            newHiraganaLetter = random.choice(myHiragana[ord(letter)])
            if random.random() < 0.3:
                newLatinLetter = rotate(newLatinLetter)
                newHiraganaLetter = rotate(newHiraganaLetter)
            if random.random() < 0.3:
                newLatinLetter = stretch(newLatinLetter)
                newHiraganaLetter = stretch(newHiraganaLetter)
            newLatinLetter = resize(newLatinLetter)
            newHiraganaLetter = resize(newHiraganaLetter)
            latinLine[0:32, index:index+32] = newLatinLetter
            hiraganaLine[0:32, index:index + 32] = newHiraganaLetter
        index += 32


    printLine = (currentLine - (114 * pagesNo))

    latinPage[(printLine * 32):((printLine * 32) + 32), 0:(32 * 80)] = latinLine
    hiraganaPage[(printLine * 32):((printLine * 32) + 32), 0:(32 * 80)] = hiraganaLine

    currentLine += 1

latinPage =to_binary(latinPage)
mpl.image.imsave(f'data/latin/latin{pagesNo}.png', latinPage, cmap='gray')
hiraganaPage =to_binary(hiraganaPage)
mpl.image.imsave(f'data/hiragana/hiragana{pagesNo}.png', hiraganaPage, cmap='gray')

