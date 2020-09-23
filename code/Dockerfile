FROM ubuntu:18.04

## Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        wget \
        nano \
        python3-pip \
        software-properties-common


RUN pip3 install DateTime requests numpy pandas==1.1.2 lightgbm category-encoders pickle-mixin

WORKDIR /work

# copy entire directory where docker file is into docker container at /work
COPY . /work/

RUN chmod 777 train.sh
RUN chmod 777 test.sh
