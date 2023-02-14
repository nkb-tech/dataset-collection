ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=20.04
ARG CUDNN_VERSION=8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION} AS main

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install \
        --yes \
        --no-install-recommends \
            build-essential \
            ffmpeg \
            libsm6 \
            libxext6 \
            git \
            ca-certificates \
            lsb-release \
            sudo \
            gnupg2 \
            python3-pip \
            python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ARG DOCKER_USER
ARG DOCKER_UID
ARG DOCKER_GID
ARG DOCKER_PW
ARG DEFAULT_PATH=/home
ARG PROJECT_PATH=src/
ARG PROJECT_NAME=project

RUN useradd \
        --create-home ${DOCKER_USER} \
        --uid=${DOCKER_UID} \
    && echo "${DOCKER_USER}:${DOCKER_PW}" | chpasswd \
    && adduser ${DOCKER_USER} sudo \
    && usermod -a -G sudo ${DOCKER_USER}

USER ${DOCKER_UID}:${DOCKER_GID}

WORKDIR ${DEFAULT_PATH}/${DOCKER_USER}

COPY --chown=${DOCKER_USER} requirements.txt ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/
COPY --chown=${DOCKER_USER} requirements/ ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements/

# install default requirements
RUN pip install \
    --no-cache-dir \
    --requirement \
        ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements/preinstall.txt \
    --requirement \
        ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements/install.txt \
    && rm ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements.txt \
    && rm -rf ${DEFAULT_PATH}/${DOCKER_USER}/${PROJECT_PATH}/${PROJECT_NAME}/requirements/

# install mmlab stuff directly
# (because of we need configs)
# TODO: rm this and add to requirements
RUN git clone https://github.com/open-mmlab/mmdetection.git ${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmdet \
    && cd ${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmdet \
    && pip install \
        --no-cache-dir \
        --verbose \
        --editable . \
    && git clone https://github.com/open-mmlab/mmpose.git ${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmpose \
    && cd ${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmpose \
    && pip install \
        --no-cache-dir \
        --verbose \
        --editable .

ENV PYTHONPATH="${PYTHONPATH}:${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmpose:${DEFAULT_PATH}/${DOCKER_USER}/mmlab/mmdet"
