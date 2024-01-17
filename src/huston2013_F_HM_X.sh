#!/usr/bin/env bash

#training the fusion model of hsi and ms

python huston2013_multi_train.py '../data/huston2013' 'hsi+ms' 0 0
python huston2013_multi_train.py '../data/huston2013' 'hsi+ms' 0 1
python huston2013_multi_train.py '../data/huston2013' 'hsi+ms' 0 2
