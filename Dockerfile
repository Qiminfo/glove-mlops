FROM nvidia/cuda:10.2-runtime-ubuntu18.04 AS base
RUN apt-get -qq update \
        && apt-get -qq install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev
WORKDIR /py-build
ENV PY_VERSION=3.8.8
RUN wget https://www.python.org/ftp/python/${PY_VERSION}/Python-${PY_VERSION}.tgz
RUN tar -xf Python-${PY_VERSION}.tgz
WORKDIR /py-build/Python-${PY_VERSION}
RUN ./configure --enable-optimizations
RUN make -j $(nproc)
RUN make install
RUN python3.8 --version
RUN apt-get -qq install python3-pip
RUN python3.8 -m pip install pip
WORKDIR /workspace
ENV PYTHONPATH /workspace
COPY pip.conf /etc/
COPY requirements.txt .
RUN python3.8 -m pip install -r requirements.txt

FROM base AS nlp
RUN apt-get install git -y
COPY requirements-nlp.txt .
RUN python3.8 -m pip install -r requirements-nlp.txt
RUN python3.8 -c "import nltk; nltk.download('stopwords')"
COPY glove /workspace/glove/
RUN cd glove \
       && python3.8 setup.py build \
       && python3.8 setup.py install \
       && cd .. \
       && python3.8 -c "from glove import Glove"

FROM nlp AS ci
COPY requirements-ci.txt .
RUN python3.8 -m pip install -r requirements-ci.txt
COPY . .
RUN chmod 755 ./demo.sh
