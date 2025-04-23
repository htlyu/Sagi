#!/bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
IMAGE_NAME="registry-intl.cn-hongkong.aliyuncs.com/flavius/chatbot-python:latest"

build_docker_image() {
    debug=""
    # uncomment to show progress in plain mode for debugging
    debug="--progress=plain"

    # clear old image if exists
    docker image rmi ${IMAGE_NAME} 2>/dev/null || true

    # build new image
    docker buildx build ${debug} -t ${IMAGE_NAME} ${root_dir} -f ${root_dir}/dev/Dockerfile
}

run_docker_container() {
    username=$(whoami)
    echo mounting $root_dir to /chatbot in container
    docker run -itd --rm --name ${username}_chatbot_open \
        -v "$root_dir:/chatbot" \
        --cap-add=SYS_ADMIN \
        ${IMAGE_NAME}
    echo you may run \"docker exec -it "${username}_chatbot_open" /bin/bash\" to enter the container
}

set -e
### on host:
build_docker_image
run_docker_container
