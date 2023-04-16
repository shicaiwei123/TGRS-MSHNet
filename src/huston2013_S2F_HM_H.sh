#!/usr/bin/env bash
python huston2013_shared_specific_fusion.py '../data/huston2013' 'ms+hsi' 'cross_patch_kd_jda_avg_fc_hsi+ms_hsi_lr_0.001_version_2.pth' 'single_fc_modal_hsi_version_2.pth' 1 0
python huston2013_shared_specific_fusion.py '../data/huston2013' 'ms+hsi' 'cross_patch_kd_jda_avg_fc_hsi+ms_hsi_lr_0.001_version_2.pth' 'single_fc_modal_hsi_version_2.pth' 1 1