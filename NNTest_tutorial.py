from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from numpy import load, zeros
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



# Dataset preppings

path = 'dataset'

# Get files to add
files = os.listdir(path)
n_files = len(files)

# Init array
dataset = zeros((n_files, 15, 14))

# x - n_files
# y - mfcc
# z - value

# Filling in the 3D array
for i in range(0, n_files):

    # Load current file
    file = Path(path) / files[i]
    array = load(file)

    dataset[i] = array

# Split into x and y
x = dataset[0:dataset.shape[0], 0:15, 0:13]
y = dataset[0:dataset.shape[0], 0:15, [13]]



# # Splitting the DF to get 80/20 elements
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= (0.2), random_state=1, shuffle=True)

# Each file has 15 seconds == 15 samples
height = 15

# 13 MFCC per sample
width = 13

# 1 plane
depth = 1

input_shape = (height, width, depth)



x_train = x_train.reshape(x_train.shape[0], height, width, depth)
x_test = x_test.reshape(x_test.shape[0], height, width, depth)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(height, width, depth)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=1, validation_data=(x_test, y_test))