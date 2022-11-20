# Tensorflow / Keras
from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Model # for creating a Neural Network Autoencoder model
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, LeakyReLU, BatchNormalization # for adding layers to DAE model
from tensorflow.keras.utils import plot_model # for plotting model diagram

from sklearn.model_selection import train_test_split
# Data manipulation
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version

# Visualization
import matplotlib
import matplotlib.pyplot as plt # for plotting model loss
print('matplotlib: %s' % matplotlib.__version__) # print version
import graphviz # for showing model diagram
print('graphviz: %s' % graphviz.__version__) # print version

# Other utilities
import sys
import os

# Assign main directory to a variable
main_dir=os.path.dirname(sys.path[0])
#print(main_dir)

clean = np.load('datasets/denoiseAE/withoutNoise.npy')
noised = np.load('datasets/denoiseAE/withNoise.npy')

clean = clean.reshape(clean.shape[0], 32*32)
noised = noised.reshape(noised.shape[0], 32*32)

train = 40_000
test = 4_000

cleanTrain = clean[:train]
cleanTest = clean[train:train+test]

noisedTrain = noised[:train]
noisedTest = noised[train:train+test]


#--- Define Shapes
n_inputs=cleanTrain.shape[1] # number of input neurons = the number of features X_train

#--- Input Layer
visible = Input(shape=(n_inputs,), name='Input-Layer') # Specify input shape

#--- Encoder Layer
e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
e = LeakyReLU(name='Encoder-Layer-Activation')(e)

#--- Middle Layer
middle = Dense(units=n_inputs, activation='linear', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer')(e)

#--- Decoder Layer
d = Dense(units=n_inputs, name='Decoder-Layer')(middle)
d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
d = LeakyReLU(name='Decoder-Layer-Activation')(d)

#--- Output layer
output = Dense(units=n_inputs, activation='sigmoid', name='Output-Layer')(d)

# Define denoising autoencoder model
model = Model(inputs=visible, outputs=output, name='Denoising-Autoencoder-Model')

# Compile denoising autoencoder model
model.compile(optimizer='adam', loss='mse')

# Print model summary
print(model.summary())



history = model.fit(noisedTrain, cleanTrain, epochs=20, batch_size=32, verbose=1, validation_data=(noisedTest, cleanTest))

# Plot a loss chart
fig, ax = plt.subplots(figsize=(16,9), dpi=300)
plt.title(label='Model Loss by Epoch', loc='center')

ax.plot(history.history['loss'], label='Training Data', color='black')
ax.plot(history.history['val_loss'], label='Test Data', color='red')
ax.set(xlabel='Epoch', ylabel='Loss')
plt.xticks(ticks=np.arange(len(history.history['loss'])), labels=np.arange(1, len(history.history['loss'])+1))
plt.legend()

plt.show()


# Apply denoising model
#X_train_denoised = model.predict(X_train_noisy).reshape(60000, 28, 28)
X_test_denoised = model.predict(noisedTest).reshape(noisedTest.shape[0], 32, 32)
cleanTest = cleanTest.reshape(cleanTest.shape[0], 32, 32)
noisedTest = noisedTest.reshape(noisedTest.shape[0], 32, 32)
# Display images of the first 10 digits
fig, axs = plt.subplots(6, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
for i in range(0,2):
    for j in range(0,5):
        axs[i, j].matshow(cleanTest[n])
        axs[i + 2, j].matshow(noisedTest[n])
        axs[i+4,j].matshow(X_test_denoised[n])
        n=n+1
plt.show()