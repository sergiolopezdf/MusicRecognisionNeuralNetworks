from numpy import load, zeros
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def get_dataset():
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

    # Splitting the DF to get 80/20 elements
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

    # Each file has 15 seconds == 15 samples
    height = 15

    # 13 MFCC per sample
    width = 13

    # 1 plane
    depth = 1

    input_shape = (height, width, depth)

    # Reshape to fit Keras dimensions
    x_train = x_train.reshape(x_train.shape[0], height, width, depth)
    x_test = x_test.reshape(x_test.shape[0], height, width, depth)
    y_train = y_train.reshape(y_train.shape[0], height)
    y_test = y_test.reshape(y_test.shape[0], height)

    # Setting labels
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # We need just a label per 15 MFCC
    y_train = y_train[0:, [0], 0:]
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
    y_test = y_test[0:, [0], 0:]
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

    # By making hot encoding, it creates an array with 1/0, being 1 when the tag is correct

    return x_train, x_test, y_train, y_test, input_shape
