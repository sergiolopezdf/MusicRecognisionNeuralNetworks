from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential


def get_model(input_shape):
    model = Sequential()

    # 1st block
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))

    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

    model.add(Dropout(0.25))

    # 2nd block
    model.add(Conv2D(64, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

    model.add(Dropout(0.25))

    # 3rd block
    model.add(Conv2D(128, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

    model.add(Dropout(0.25))

    # 4th block
    model.add(Conv2D(256, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(Conv2D(256, (3, 3), padding="same"))

    model.add(Activation("relu"))

    model.add(GlobalMaxPooling2D())

    # Flatten
    model.add(Flatten())

    model.add(Dropout(0.5))

    # Last layer
    model.add(Dense(2, activation="softmax"))

    print(model.summary())

    return model
