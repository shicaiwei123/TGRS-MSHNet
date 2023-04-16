#!/usr/bin/env bash
python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+ms' 'cross_patch_kd_jda_avg_fc_hsi+ms_ms_lr_0.001_version_3.pth' 'MDL_single_modal_ms.pth' 0 0
python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+ms' 'cross_patch_kd_jda_avg_fc_hsi+ms_ms_lr_0.001_version_3.pth' 'MDL_single_modal_ms.pth' 0 1
python huston2013_shared_specific_fusion.py '../data/huston2013' 'hsi+ms' 'cross_patch_kd_jda_avg_fc_hsi+ms_ms_lr_0.001_version_3.pth' 'MDL_single_modal_ms.pth' 0 2