#!/bin/bash

salloc -N1 --mem-per-gpu=12G -t00:30:00 --gres=gpu:V100:1 --ntasks-per-node=1
