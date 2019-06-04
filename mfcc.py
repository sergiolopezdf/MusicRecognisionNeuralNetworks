import librosa
from numpy import save, array
import os
from pathlib import Path


def read_folder(folder_path, output_folder):
    files = os.listdir(folder_path)
    duration = 5

    for track in files:
        path = Path(folder_path) / track

        # Getting the metadata for tagging
        metadata = track.split('_')
        tag = int(metadata[0])
        offset = int(metadata[1])
        name = metadata[2].split('.')[0]

        # Audio loading + MFCC computation
        x, fs = librosa.load(path, offset=offset, duration=duration)
        mfccs = librosa.feature.mfcc(x, sr=fs, n_mfcc=13)

        # Adding the tag
        mfccs_tag = array([mfccs, tag])

        # Adding proper naming to the output file
        output_name = str(tag) + '_' + str(offset) + '_' + name
        output_file = Path(output_folder) / output_name

        # Save analyzed tracks
        save(output_file, arr=mfccs_tag)


def get_mfcc_single_track(track_path, offset):
    track_path = Path(track_path)

    # Audio loading + MFCC computation
    x, fs = librosa.load(track_path, offset=offset, duration=5)
    mfccs = librosa.feature.mfcc(x, sr=fs, n_mfcc=13)

    return mfccs


def get_duration(track_path):
    path = Path(track_path)
    y, sr = librosa.load(path)
    return librosa.get_duration(y=y, sr=sr)





# read_folder('files', 'dataset_new')
