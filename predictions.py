from tensorflow.keras.models import load_model
from mfcc import get_mfcc_single_track, get_duration
from dataset import prepare_single_track
from pathlib import Path

model = load_model('test_model.h5')
file = 'test/HunterRmx.mp3'


def analyze_full_track(path):
    duration = get_duration(Path(file))
    duration = int(duration)
    min = 0
    seg = 0

    for i in range(5, duration, 5):
        offset = 60 * min + seg

        mfccs = get_mfcc_single_track(file, offset)

        mfccs = prepare_single_track(mfccs)

        pred = model.predict(mfccs)

        result = pred.argmax()
        print(str(min) + ':' + str(seg) + ' = ' + str(result))

        seg = i % 60
        min = int(i / 60)


analyze_full_track(file)
