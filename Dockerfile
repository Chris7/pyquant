FROM ubuntu:14.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    gfortran \
    git \
    libxml2-dev \
    libxslt1-dev \
    python-dev \
    python-numpy \
    zlib1g-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o - | python

RUN pip install cython
RUN pip install --upgrade setuptools

RUN git clone https://github.com/Chris7/pyquant.git

WORKDIR pyquant

RUN pip install .

ENTRYPOINT ["pyQuant"]
CMD ["--help"]
