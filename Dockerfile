FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

FROM continuumio/miniconda3

SHELL ["/bin/bash", "--login", "-c"]

COPY conda_environment.yml .
RUN conda env create -f conda_environment.yml

SHELL ["conda", "run", "-n", "vertical-cas-env", "/bin/bash", "-c"]
RUN echo "conda activate vertical-cas-env" > ~/.bashrc

RUN apt-get -y update && apt-get install -y git gcc g++ cmake

RUN pip install --upgrade pip cvxpy pyinterval

# Can thin this out to lessen loading times
# Have Marabou in here, rather than cloning and building it
COPY . .

RUN git clone https://github.com/NeuralNetworkVerification/Marabou.git \
    && mv Marabou GenerateNetworks/Marabou \
    && cd GenerateNetworks/Marabou && mkdir build && cd build && cmake .. -DBUILD_PYTHON=ON \
    && cmake --build .

ENV PYTHONPATH=$PYTHONPATH:$(pwd)/GenerateNetworks/Marabou


CMD echo "running test" && python3 GenerateNetworks/trainVertCAS.py
