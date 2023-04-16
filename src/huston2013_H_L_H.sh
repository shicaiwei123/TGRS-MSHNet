#!/usr/bin/env bash
python huston2013_hall_infer_ensemble.py '../data/huston2013' 'lidar+hsi' 'single_to_single_vkd_mse_fc_lidar_hsi_lr_0.001_version_1.pth' 'single_fc_modal_hsi_version_2.pth' 1 0
python huston2013_hall_infer_ensemble.py '../data/huston2013' 'lidar+hsi' 'single_to_single_vkd_mse_fc_lidar_hsi_lr_0.001_version_2.pth' 'single_fc_modal_hsi_version_2.pth' 1 1