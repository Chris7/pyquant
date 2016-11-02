FROM ubuntu:14.04

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
    zlib1g-dev \
    git

RUN pip install numpy scipy pandas cython

RUN git clone https://github.com/Chris7/pyquant.git

WORKDIR pyquant

RUN pip install .

ENTRYPOINT ["pyQuant"]
CMD ["--help"]
