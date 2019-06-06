import librosa
from numpy import save, array
import os
from pathlib import Path

import subprocess


# ffmpeg -i 0_65_MujerBruja.mp3 -af lowpass=10000,highpass=300 filtered.mp3


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

        # Adding proper naming to the output file
        output_name = str(tag) + '_' + str(offset) + '_' + name
        output_file = Path(output_folder) / output_name

        if os.path.isfile(output_folder + '/' + output_name + '.npy'):
            continue

        # Audio loading + MFCC computation
        x, fs = librosa.load(path, offset=offset, duration=duration)
        mfccs = librosa.feature.mfcc(x, sr=fs, n_mfcc=13)

        # Adding the tag
        mfccs_tag = array([mfccs, tag])

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


def filter_all(path, output_filtered):
    files = os.listdir(path)

    for track in files:
        output = Path(output_filtered) / track

        if os.path.isfile(output):
            continue

        track_path = Path(path) / track

        command = ['ffmpeg', '-i', Path(track_path), '-af', 'lowpass=10000,highpass=300', output]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        print(track + ' has been filtered properly')


def filter_one(track):
    path = Path(track)
    command = ['ffmpeg', '-i', Path(path), '-af', 'lowpass=10000,highpass=300', 'tmp.mp3', '-y']
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
    print(track + ' has been filtered properly')
