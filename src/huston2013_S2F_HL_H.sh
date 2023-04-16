#!/usr/bin/env bash


python huston2013_shared_specific_fusion.py '../data/huston2013' 'lidar+hsi' 'jda_hsi+lidar_hsi_lr_0.001_version_1.pth' 'MDL_single_modal_hsi.pth' 1 0
python huston2013_shared_specific_fusion.py '../data/huston2013' 'lidar+hsi' 'jda_hsi+lidar_hsi_lr_0.001_version_1.pth' 'MDL_single_modal_hsi.pth' 1 1
python huston2013_shared_specific_fusion.py '../data/huston2013' 'lidar+hsi' 'jda_hsi+lidar_hsi_lr_0.001_version_1.pth' 'MDL_single_modal_hsi.pth' 1 2