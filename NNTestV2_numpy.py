from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
from numpy import load, array, vstack, empty, append
from pathlib import Path
import os
from sklearn.model_selection import train_test_split



# Dataset preppings

path = 'dataset'

files = os.listdir(path)

dataset = load(Path(path) / 'init.npy')


for mfcc_file in files:

    # If it's init file, skip it
    if mfcc_file == 'init.npy':
        continue

    # Load current file
    file = Path(path) / mfcc_file
    array = load(file)

    # Init has this dimension. If dataset shape is 1, 14; it's empty
    s = (1, 14)

    if dataset.shape == s:
        dataset = array
        continue

    dataset = vstack((dataset, array))

print(dataset)
#
# x = []
# y = []

# for index in dataset:
#     x.append(index[0:12])
#     y.append(index[13])
#
# # To numpy
# x = array(x)
# y = array(y)
#
# print(x)




#
# # # Splitting the DF to get 80/20 elements
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= (0.2), random_state=1, shuffle=True)
#
# height = 99
# width = 12
# depth = 1
#
# input_shape = (height, width, depth)
#
#
# x_train = x_train.reshape(1, 99, 12, 1)
# x_test = x_test.reshape(1, 25, 12, 1)
#
# model = Sequential()
#
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
#
# model.add(Activation("relu"))
#
# model.add(Conv2D(32, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(Conv2D(64, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Dropout(0.25))
#
# model.add(Conv2D(128, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(Conv2D(128, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(MaxPooling2D(pool_size=(3, 3)))
#
# model.add(Dropout(0.25))
#
# model.add(Conv2D(256, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(Conv2D(256, (3, 3), padding="same"))
#
# model.add(Activation("relu"))
#
# model.add(GlobalMaxPooling2D())
#
# model.add(Flatten())
#
# model.add(Dropout(0.5))
#
# model.add(Activation("softmax"))
#
# model.fit(x_train, y_train,
#           epochs=2,
#           validation_data=(x_test, y_test),
#           )
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print(model.summary())
