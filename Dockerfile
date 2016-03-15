FROM phusion/baseimage:0.9.18

RUN apt-get update && apt-get install -y \
    libxml2-dev \
    build-essential \
    gcc \
    python-dev \
    python-pip \
    liblapack-dev \
    libblas-dev \
    gfortran \
    libxslt1-dev \
    zlib1g-dev

RUN pip install numpy scipy pandas cython

RUN pip install pyquant-ms
