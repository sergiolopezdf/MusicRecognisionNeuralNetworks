from numpy import load, zeros
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def get_dataset():
    path = 'dataset_new'

    # Get files to add
    files = os.listdir(path)
    n_files = len(files)

    file = Path(path) / files[0]

    array = load(file, allow_pickle=True)

    n_mfcc = array[0].shape[0]
    n_samples = array[0].shape[1]

    # Init array
    x = zeros((n_files, n_mfcc, n_samples))
    y = zeros((n_files))

    # x - n_files
    # y - mfcc
    # z - value

    # Filling in the 3D array
    for i in range(0, n_files):
        # Load current file

        file = Path(path) / files[i]

        array = load(file, allow_pickle=True)

        x[i] = array[0]
        y[i] = int(array[1])

    # Split into x and y

    # Splitting the DF to get 80/20 elements
    keep_prob = 0.8
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - keep_prob, random_state=1, shuffle=True)

    height = n_mfcc  # 13 MFCC per sample
    width = n_samples
    depth = 1

    input_shape = (height, width, depth)

    # Reshape to fit Keras dimensions
    x_train = x_train.reshape(x_train.shape[0], height, width, depth)
    x_test = x_test.reshape(x_test.shape[0], height, width, depth)

    # Setting labels
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # By making hot encoding, it creates an array with 1/0, being 1 when the tag is correct
    return x_train, x_test, y_train, y_test, input_shape


def prepare_single_track(track_mfcc):
    return track_mfcc.reshape(1, track_mfcc.shape[0], track_mfcc.shape[1], 1)



get_dataset()
