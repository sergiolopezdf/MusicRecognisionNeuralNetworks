# Singer detection in music files based on AI

The aim of this project is to prove that detecting if someone is singing in a song is possible. This relies on analysing
 the MFCC of the track 

## How to install

This project has been build upon an Conda3/Anaconda3 environment. Simply import the environment with 
`conda env create -f MusicAnalyzerEnvironment.yml`

## System requirements
* Conda3/Anaconda3 
* FFMPEG
* TensorFlow
* Keras
* Librosa

## How to run

### Train
Run the `train.py`file. You have to choose whether to analyze an entire audio files folder (using the `read_folder(folder_to_analyze, output_folder)` function)  or to use an already analyzed folder by commenting that function and simply completing the `dataset` variable.

The audio files must follow this sintax: 
* Positive or negative detection (0/1)
* Second to start analyzing
* Name (not critical)

For instance, if you want to read a snippet of a track starting in 00:07 which has a singer on it, the scheme would be  as follows: 0_7_RandomName.mp3

The system reads 5 seconds of the track. This can be changed by diving into the code 

### Analyze a full track

Run the `run.py` file. Simply select the model you want to use and the path to the file you want to read