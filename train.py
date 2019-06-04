from model import get_model
from dataset import get_dataset

# Get values
x_train, x_test, y_train, y_test, input_shape = get_dataset()

# Get model
model = get_model(input_shape)

# Compile model. Categorical_crossentropy since Softmax
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=80,
          validation_data=(x_test, y_test),
          )

# Tests
score = model.evaluate(x_test, y_test, verbose=0)

model.save('test_model.h5')

print('Test loss:', score[0])
print('Accuracy: ', score[1])

print(x_test[0:1].shape)

# print(model.predict(x_test[0:1]))
