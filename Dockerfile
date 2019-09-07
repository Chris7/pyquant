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
    zlib1g-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o - | python

RUN pip install cython
RUN pip install --upgrade setuptools

WORKDIR pyquant
COPY Makefile MANIFEST.in requirements.txt setup.py tox.ini ./
RUN pip install -r requirements.txt

COPY scripts scripts
COPY tests tests
COPY pyquant pyquant

RUN pip install -e .

ENTRYPOINT ["pyQuant"]
CMD ["--help"]
