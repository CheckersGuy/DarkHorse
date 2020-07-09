FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
      apt-get -y install sudo
RUN apt-get -y install libtbb-dev
RUN apt-get -y install build-essential
RUN apt-get -y install cmake
RUN apt-get -y install libprotobuf-dev
RUN apt-get -y install protobuf-compiler
RUN useradd -ms /bin/bash robin
WORKDIR /home/robin
RUN mkdir DarkHorse
RUN mkdir Dokumente && cd Dokumente && mkdir CWeights
COPY Checkers DarkHorse/Checkers
COPY CMakeLists.txt DarkHorse/CMakeLists.txt
COPY Training DarkHorse/Training
WORKDIR /home/robin/DarkHorse
RUN cmake CMakeLists.txt && make -j ${nproc}
RUN adduser robin sudo
USER robin
ARG HOME=/home/robin

