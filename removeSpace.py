import numpy as np
import matplotlib.pyplot as plt
import random

''' Odkomentuj potrzebny dataset'''
# type = "hiragana"
type = "latin"


dataset = np.load('data/ourDataset_noise0.07_lines.npy')

def remove_space(datasets, type, coef = 0.14):
    space = []
    mark = []
    number = 0 if type == "latin" else 1
    for i in range(len(datasets)):
        value, count = np.unique(datasets[i][number], return_counts= True)
        # print(count, value, count[0]/count[-1] )
        if count[0]/count[-1] < coef:
            space.append(datasets[i][number])
        else:
            mark.append(datasets[i][number])
        # plt.imshow(dataset[i][number], cmap='binary')
        # plt.show()
    return np.array(mark), np.array(space)

latin, latin_space = remove_space(dataset, type, coef = 0.14)
type = "hiragana"
hiragana, hiragana_space = remove_space(dataset, type, coef = 0.14)