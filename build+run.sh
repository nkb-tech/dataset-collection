#!/bin/bash
export DOCKER_UID
export DOCKER_IMAGE_NAME
export DOCKER_IMAGE_VERSION
export DOCKER_USER_NAME
export DOCKER_USER_PASSWORD
export DOCKER_PROJECT_PATH
export DOCKER_PROJECT_NAME
export DOCKER_DEFAULT_PATH
export DOCKER_CONTAINER_NAME

docker build . \
    --file Dockerfile \
    --tag ${DOCKER_IMAGE_NAME}/${DOCKER_IMAGE_VERSION} \
    --build-arg UID=${DOCKER_UID} \
    --build-arg GID=${DOCKER_UID} \
    --build-arg PW=${DOCKER_USER_PASSWORD} \
    --build-arg USER=${DOCKER_USER_NAME} \
    --build-arg PROJECT_PATH=${DOCKER_PROJECT_PATH} \
    --build-arg PROJECT_NAME=${DOCKER_PROJECT_NAME} \
    --build-arg DEFAULT_PATH=${DOCKER_DEFAULT_PATH}

docker run \
    -itd \
    --ipc host \
    --gpus all \
    --name ${DOCKER_CONTAINER_NAME} \
    --net "host" \
    --env "DISPLAY" \
    --env "QT_X11_NO_MITSHM=1" \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume $PWD:${DOCKER_DEFAULT_PATH}/${DOCKER_USER_NAME}/${DOCKER_PROJECT_PATH}/${DOCKER_PROJECT_NAME}/:rw \
    --privileged \
    ${DOCKER_IMAGE_NAME}/${DOCKER_IMAGE_VERSION}

docker start ${DOCKER_CONTAINER_NAME}
docker exec \
    -it ${DOCKER_CONTAINER_NAME} \
    bash -c "cd ${DOCKER_DEFAULT_PATH}/${DOCKER_USER_NAME}/${DOCKER_PROJECT_PATH}/${DOCKER_PROJECT_NAME}; bash"
