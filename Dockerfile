FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

FROM continuumio/miniconda3

SHELL ["/bin/bash", "--login", "-c"]

COPY conda_environment.yml .
RUN conda env create -f conda_environment.yml

SHELL ["conda", "run", "-n", "vertical-cas-env", "/bin/bash", "-c"]
RUN echo "conda activate vertical-cas-env" > ~/.bashrc

RUN apt-get -y update
RUN apt-get install -y git

COPY . .

RUN python3 GenerateNetworks/trainVertCAS.py
