#!/usr/bin/env bash
#'''
#training the fusion model of hsi and lidar
#'''
python huston2013_multi_train.py '../data/huston2013' 'hsi+lidar' 0
python huston2013_multi_train.py '../data/huston2013' 'hsi+lidar' 1
python huston2013_multi_train.py '../data/huston2013' 'hsi+lidar' 2
