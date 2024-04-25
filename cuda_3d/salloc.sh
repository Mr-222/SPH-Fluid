#!/bin/bash

salloc -N1 --mem-per-gpu=20G -t01:00:00 --gres=gpu:H100:1 --ntasks-per-node=1 -c16
