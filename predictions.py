from tensorflow.keras.models import load_model
from dataset import push_tag

model = load_model('test_model.h5')

file = 'test/1_Milkshake.npy'

track = push_tag(file)

pred = model.predict(track)

result = pred.argmax()
print(result)
