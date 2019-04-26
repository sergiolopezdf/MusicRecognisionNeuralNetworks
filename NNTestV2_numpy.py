from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
from numpy import load, array, vstack, empty, append, dstack, hstack, concatenate, zeros
from pathlib import Path
import os
from sklearn.model_selection import train_test_split



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


model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))

model.add(Activation("relu"))

model.add(Conv2D(32, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(Conv2D(64, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(Conv2D(128, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(Conv2D(256, (3, 3), padding="same"))

model.add(Activation("relu"))

model.add(GlobalMaxPooling2D())

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Activation("softmax"))

model.fit(x_train, y_train,
          epochs=2,
          validation_data=(x_test, y_test),
          )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.summary())
