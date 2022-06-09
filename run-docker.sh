#!/bin/bash
nvidia-docker run --rm -it -e NVIDIA_VISIBLE_DEVICES=0,1 --shm-size=1g --ulimit memlock=-1 --mount type=bind,src=$PWD,dst=/gpt-neox -v $PWD/../20B_checkpoints:/gpt-neox/20B_checkpoints gpt-neox-oom