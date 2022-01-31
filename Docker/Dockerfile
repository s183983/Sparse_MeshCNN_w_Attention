FROM pytorch/conda-cuda:latest
RUN export CPATH="usr/local/cuda/include"
RUN git clone https://github.com/ThomasOerkild/MeshCNN.git
RUN conda env create -f MeshCNN/environment.yml
RUN conda init bash

