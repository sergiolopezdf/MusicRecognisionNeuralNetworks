from tensorflow.keras.models import load_model
from predictions import analyze_full_track

# Set model to use
model = load_model('model_0606.h5')

# Set file to analyze
file = 'test/HunterRmx.mp3'

# Analyze full track
analyze_full_track(file, model)
