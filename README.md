# MSHNet
Code for MSH-Net: Modality-Shared Hallucination with Joint Adaptation Distillation for Remote Sensing Image Classification Using Missing Modalities

## Dependency
- Ubuntu20.04
- CUDA11.3
- PyToch1.12
- python3.8

## Dataset
- Download
  - Original [Huston20013](https://github.com/danfenghong/ISPRS_S2FL), [Augsburg](https://github.com/danfenghong/ISPRS_S2FL)
  - Preprocessing [Huston2013](https://drive.google.com/drive/folders/1YSbAFzD9MKcNMBbYTeax_c1XNkSjZC_a) [Augsburg](https://drive.google.com/drive/folders/1f4bvCefoJ9Xd6QTbByDSBY5x7pAW1u2q)
  - precessing code: https://github.com/danfenghong/IEEE_TGRS_GCN/tree/master/DataGeneration_Functions
- Build soft link
  ```bash
  cd MSHNET
  mkdir data
  ln -s path_to_download_data ./data/dataset_name
  
  for example: ln -s /home/data/shicaiwei/remote_sensing/huston2013 ./data/huston2013
  ```


## Train

### Name Rules of Bash Files 
- dataset_operation_modality_.sh

  - dataset
    - huston2013
    - Augsburg
  - operation
    - S: single modality model training
    - F: Fusing multimodal data
    - T: transfer modality-shared knowledge
    - S2F: modality share and specific fusion
  - modality
    - H: HSI modality
    - S: Sar modality
    - L: LiDAR modality
    - M: MS modality
    - D: DSM modality
- example
  - huston2013_S_H_X.sh
  - training the single modality model with HSI modality 

### Train process
  - To average the results, for each sub-task, we train three models and choose the one with middle performance for the following task. 

### Train the single modality baseline
```bash
cd src
bash huston2013_S_L_X.sh
bash huston2013_S_M_X.sh
bash huston2013_S_H_X.sh
bash augsburg_S_H_X.sh
bash augsburg_S_S_X.sh
bash augsburg_S_D_X.sh
```

### Train multimodal teacher
```bash
cd src
bash huston2013_F_HL_X.sh
bash huston2013_F_HM_X.sh
bash augsburg_F_HSD_X.sh
```

### Modality-shared hallucination

```bash
cd src
bash huston2013_T_HM_H.sh
bash huston2013_T_HM_M.sh
bash huston2013_T_HL_L.sh
bash huston2013_T_HL_H.sh
bash augsburg_T_HS_H.sh
bash augsburg_T_HS_L.sh
bash augsburg_T_HSD_H.sh
bash augsburg_T_HSD_S.sh
bash augsburg_T_HSD_D.sh
```

### Fusion modality-shared and specific information

```bash
cd src
bash huston2013_S2F_HL_L.sh
bash huston2013_S2F_HL_H.sh
bash huston2013_S2F_HM_M.sh
bash huston2013_S2F_HM_H.sh
bash augsburg_S2F_HS_HS_S.sh
bash augsburg_S2F_HS_HS_H.sh
bash augsburg_S2F_HSD_H.sh
bash augsburg_S2F_HSD_S.sh
bash augsburg_S2F_HSD_D.sh

```

