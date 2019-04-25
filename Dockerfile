FROM continuumio/miniconda3
RUN apt-get update
WORKDIR home/
RUN git clone https://github.com/sergiolopezdf/MusicRecognisionNeuralNetworks
WORKDIR MusicRecognisionNeuralNetworks
RUN conda env create -f neuralNetworksEnv.yml