#!/bin/bash

# $1 ... experimental tag used for screen
# $2 ... experimental call

for fold in 1 2 3 4
do
    gpu=$((fold - 1))
    cmd="THEANO_FLAGS=\"device=cuda$gpu\" $2 --fold $fold"
    screen_name="$1_$fold"
    echo $cmd
    screen -d -m -S "$screen_name" bash -c "$cmd"
done
