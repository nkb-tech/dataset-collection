ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install \
        --yes \
        --no-install-recommends \
            build-essential \
            ffmpeg \
            libsm6 \
            libxext6 \
            wget \
            curl \
            git \
            vim \
            cmake \
            ca-certificates \
            lsb-release \
            sudo \
            gnupg2 \
            zip \
            unzip \
            python3-pip \
            python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ARG USER=user
ARG UID=1000
ARG GID=1000
ARG PW=user
ARG DEFAULT_PATH=/home
ARG PROJECT_PATH=src/
ARG PROJECT_NAME=project

WORKDIR ${DEFAULT_PATH}/${USER}

RUN useradd -m ${USER} --uid=${UID} \
    && echo "${USER}:${PW}" | chpasswd \
    && adduser ${USER} sudo \
    && usermod -a -G sudo ${USER}

RUN mkdir -p ${DEFAULT_PATH}/${USER}/${PROJECT_PATH}/ && chown -R ${UID}:${GID} ./
COPY --chown=${USER}:${USER} requirements.txt ${DEFAULT_PATH}/${USER}/${PROJECT_PATH}/${PROJECT_NAME}/

# install default requirements
RUN python3 -m pip install \
        --no-cache-dir \
        --upgrade \
            pip \
            wheel \
            setuptools \
            packaging \
    && python3 -m pip install \
        --no-cache-dir \
        --ignore-installed \
        --requirement \
            ${DEFAULT_PATH}/${USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements.txt \
    && rm ${DEFAULT_PATH}/${USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements.txt

# install mmlab stuff directly
# (because of we need configs)
RUN git clone https://github.com/open-mmlab/mmdetection.git ${DEFAULT_PATH}/${USER}/mmlab/mmdet \
    && cd ${DEFAULT_PATH}/${USER}/mmlab/mmdet \
    && pip install \
        --no-cache-dir \
        --verbose \
        --editable . \
    && git clone https://github.com/open-mmlab/mmpose.git ${DEFAULT_PATH}/${USER}/mmlab/mmpose \
    && cd ${DEFAULT_PATH}/${USER}/mmlab/mmpose \
    && pip install \
        --no-cache-dir \
        --verbose \
        --editable .

ENV PYTHONPATH="${PYTHONPATH}:${DEFAULT_PATH}/${USER}/mmlab/mmpose:${DEFAULT_PATH}/${USER}/mmlab/mmdet"
