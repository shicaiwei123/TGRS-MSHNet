#!/usr/bin/env bash


python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+lidar' 'jda_hsi+lidar_lidar_lr_0.001_version_0.pth' 'MDL_single_modal_lidar.pth' 1 0
python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+lidar' 'jda_hsi+lidar_lidar_lr_0.001_version_0.pth' 'MDL_single_modal_lidar.pth' 1 1
python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+lidar' 'jda_hsi+lidar_lidar_lr_0.001_version_0.pth' 'MDL_single_modal_lidar.pth' 1 2