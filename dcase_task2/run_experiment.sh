#!/bin/bash

# $1 ... experimental tag used for screen
# $2 ... experimental call
# $3 ... number of GPUs, defaults to 4

tag="$1"
command="$2"
num_gpus="${3:-4}"
gpu=0
for fold in {1..4}; do
    cmd="THEANO_FLAGS=\"device=cuda$gpu\" $command --fold $fold"
    echo $cmd
    screen -d -m -S "${tag}_${fold}" bash -c "$cmd"
    ((gpu++))
    if [ $gpu == $num_gpus ]; then
        # wait for current screens to finish
        while screen -ls | grep --quiet -e "${tag}_[1234]"; do
            sleep 10
        done
        gpu=0
    fi
done
