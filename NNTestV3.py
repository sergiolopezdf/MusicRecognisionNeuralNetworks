from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import reshape
from numpy import ndarray

# Dataset preppings
dataset = pd.read_json('mfcc.json', orient='index')
dataset = dataset[sorted(dataset.columns)]
dataset = dataset.replace('test1', 0)

ds1 = dataset[:10].values

ds2 = dataset[10:20].values

ds3 = dataset[20:30].values

ds4 = dataset[30:40].values

# dstot = pd.DataFrame([ds1, ds2, ds3, ds4], ['1', '2', '3', '4'])

ds = ndarray([ds1, ds2, ds3, ds4])

print(ds.size)


# # Splitting the DF to get 80/20 elements
# x = dstot.drop(columns=[13])
# y = dstot[13].values
# x = x.apply(pd.Series)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1,
#                                                     stratify=y)
#
# model = Sequential()
#
# x_train = x_train.values
#
# print(x_train.shape)
#
# x_train = x_train.reshape(x_train.shape[0], 10, 13, 1)
# print(x_train.shape)

# TEST NN
# model.add(
#     Conv2D(32, (3, 3), padding="same", input_shape=(10, 13, 1), data_format="channels_last", activation='relu'))
#
# # # model.add(Dense(12, input_dim=13, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(1, activation='sigmoid'))
# #
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(x_train, y_train, epochs=150, batch_size=10)
#
# # evaluate the model
# scores = model.evaluate(x_test, y_test)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
