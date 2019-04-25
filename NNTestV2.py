from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import reshape

# Dataset preppings
dataset = pd.read_json('mfcc.json', orient='index')
dataset = dataset[sorted(dataset.columns)]

x = dataset.drop(columns=[13])
y = dataset[13].values

x = x.apply(pd.Series)

# Splitting the DF to get 80/20 elements
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1,
                                                    stratify=y)
n = 10
height = 10
width = 13
depth = 1

input_shape = (n, height, width, depth)

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
