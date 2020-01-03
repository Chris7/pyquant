FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    gfortran \
    git \
    libxml2-dev \
    libxslt1-dev \
    python3-dev \
    zlib1g-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o - | python3

WORKDIR pyquant
COPY Makefile MANIFEST.in requirements.txt requirements-dev.txt setup.py setup.cfg tox.ini ./
RUN pip3 install -r requirements-dev.txt

COPY scripts scripts
COPY tests tests
COPY pyquant pyquant

RUN pip3 install -e .

ENTRYPOINT ["pyQuant"]
CMD ["--help"]
