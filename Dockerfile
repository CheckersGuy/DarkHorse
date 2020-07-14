FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
      apt-get -y install sudo
RUN apt-get -y install libtbb-dev
RUN apt-get -y install curl
RUN apt-get -y install make
RUN apt-get -y install cmake
RUN apt-get -y install clang
RUN apt-get -y install libprotobuf-dev
RUN apt-get -y install protobuf-compiler
WORKDIR /
RUN mkdir DarkHorse
COPY Checkers DarkHorse/Checkers
COPY CMakeLists.txt DarkHorse/CMakeLists.txt
COPY Training DarkHorse/Training
WORKDIR /tmp
RUN mkdir CWeights
RUN cd CWeights && curl  curl -L -o ${checkers5.weights} "https://drive.google.com/uc?export=download&id=1cjP2NK_DizqmZGNrh9IkpPrGbgpYRs5L"
WORKDIR /DarkHorse
RUN cmake CMakeLists.txt && make -j ${nproc}



