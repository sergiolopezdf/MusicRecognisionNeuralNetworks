from model import get_model
from dataset import get_dataset

# Get values
x_train, x_test, y_train, y_test, input_shape = get_dataset()

# Get model
model = get_model(input_shape)

# Compile model. Categorical_crossentropy since Softmax
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(x_train, y_train,
          epochs=100,
          validation_data=(x_test, y_test),
          )

# Tests
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score)
