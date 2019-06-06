from processing import get_mfcc_single_track, get_duration, filter_one
from dataset import prepare_single_track
from pathlib import Path


def analyze_full_track(track, model):
    track_path = Path(track)

    duration = get_duration(track_path)
    duration = int(duration)
    min = 0
    seg = 0

    for i in range(5, duration, 5):
        offset = 60 * min + seg

        mfccs = get_mfcc_single_track(track_path, offset)

        mfccs = prepare_single_track(mfccs)

        pred = model.predict(mfccs)

        result = pred.argmax()
        print(str(min) + ':' + str(seg) + ' = ' + str(result))

        seg = i % 60
        min = int(i / 60)


