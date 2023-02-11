#!/bin/bash

export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)
export DOCKER_USER_PASSWORD=user
export DOCKER_DEFAULT_PATH=/home
export DOCKER_PROJECT_PATH=src
export DOCKER_IMAGE_NAME=ilyabasharov
export DOCKER_IMAGE_VERSION=dataset_collection:v1.0.1
export DOCKER_CONTAINER_NAME=dataset_collection_test
export DOCKER_USER_NAME=${USER}
export DOCKER_PROJECT_NAME=project
export DEFAULT_DATA_PATH=/home/alexander/msbtech/petsearch/data
export DOCKER_DATA_PATH=data
