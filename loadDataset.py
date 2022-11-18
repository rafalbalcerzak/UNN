import numpy as np
import matplotlib.pyplot as plt
import random

dataset = np.load('data/ourDataset_noise0.07_lines.npy')

print(dataset.shape)

rpair = random.randrange(dataset.shape[0])
plt.imshow(dataset[rpair][0], cmap='binary')
plt.show()
plt.imshow(dataset[rpair][1], cmap='binary')
plt.show()



dataset = np.load('datasets/n0_b0_r0_s0/dataset_n0_b0_r0_s0.npy')

print(dataset.shape)

rpair = random.randrange(dataset.shape[0])
plt.imshow(dataset[rpair][0], cmap='binary')
plt.show()
plt.imshow(dataset[rpair][1], cmap='binary')
plt.show()