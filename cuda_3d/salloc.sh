#!/bin/bash

salloc -N1 --mem-per-gpu=20G -t03:00:00 --gres=gpu:H100:1 --ntasks-per-node=1
