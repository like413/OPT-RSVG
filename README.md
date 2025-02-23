# Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images
This is the offical PyTorch code for paper "Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images"  

## Contents
- [OPT-RSVG Dataset](#OPT-RSVG-Dataset)
- [LPVA Framework](#Performance-Comparison)
- [Performance Comparison](#Performance-Comparison)
- [Requirements](#Requirements)

## OPT-RSVG Dataset
The dataset contains 25,452 RS images and 48,952 image-query pairs.
![OPT-RSVG Dataset](fig/OPT-RSVG.png)
Training, validation, and test sample numbers for OPT-RSVG datasets.

| No. | Class Name         | OPT-RSVG dataset |  |  |
|-----|--------------------|-----------------|----|----|
|     |                    |Training        | Validation | Test |
| C01 | airplane           | 979             | 230        | 1142|
| C02 | ground track field | 1600         | 365        | 2066|
| C03 | tennis court       | 1093             | 284        | 1313|
| C04 | bridge             | 1699             | 452        | 2212|
| C05 | basketball court   | 1036       | 263        | 1385|
| C06 | storage tank       | 1050            | 271        | 1264|
| C07 | ship               | 1084              | 243        | 1241|
| C08 | baseball diamond   | 1477        | 361        | 1744|
| C09 | T junction         | 1663              | 425        | 2055|
| C10 | crossroad          | 1670              | 405        | 2088|
| C11 | parking lot        | 1049             | 268        | 1368|
| C12 | harbor             | 758              | 209        | 953 |
| C13 | vehicle            | 3294             | 811        | 4083|
| C14 | swimming pool      | 1128          | 308        | 1563|
| -   | Total              | **19580**         | **4895**   | **24477**|

The dataset is open source:
[Google Drive](https://drive.google.com/drive/folders/1e_wOtkruWAB2JXR7aqaMZMrM75IkjqCA?usp=drive_link),
[Baidu Netdisk](https://pan.baidu.com/s/1vitw0yc-j0uFFHRPxVdZig) 提取码: 92yk 

## LPVA Framework
![OPT-RSVG Dataset](fig/architecture.png)
The above line introduces the proposed framework of LPVA. It consists of five components: (1) Linguistic Backbone, which extracts linguistic features from referring expressions, (2) Progressive Attention module, which generates dynamic weights and biases for visual backbone conditioned on specific expressions, (3) Visual Backbone, which extracts visual features from raw images and its attention can be modified by language-adaptive weights, (4) Multi-Level Feature Enhancement Decoder, which aggregates visual contextual information to enhance the uniqueness, and (5) Localization Module, which predicts the bounding box.

## Performance Comparison
### Comparison with the SOTA methods for LPVA on the test set of OPT-RSVG

| Methods                | Venue  | Visual Encoder | Language Encoder | Pr@0.5 | Pr@0.6   | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU  | cmuIoU |
|------------------------|--------|----------------|------------------|-------|----------|-------|-------|--------|----------|-------|
| **One-stage:**         |  |  |  |  |          |  |  |  |          |  |
| ZSGNet                 | ICCV'19 | ResNet-50 | BiLSTM | 48.64 | 47.32    | 43.85 | 27.69 |  6.33 | 43.01    | 47.71 |
| FAOA                   | ICCV'19 | DarkNet-53 | BERT | 68.13 | 64.30    | 57.15 | 41.83 | \textcolor{blue}{15.33} | 58.79    | 65.20 |
| ReSC                   | ECCV'20 | DarkNet-53 | BERT | 69.12 | 64.63    | 58.20 | 43.01 | 14.85 | 60.18    | 65.84 |
| LBYL-Net               | CVPR'21 | DarkNet-53 | BERT | 70.22 | 65.39    | 58.65 | 37.54 |  9.46 | 60.57    | 70.28 |
| **Transformer-based:** |  |  |  |  |          |  |  |  |          |  |
| TransVG                | CVPR'21 | ResNet-50 | BERT | 69.96 | 64.17    | 54.68 | 38.01 | 12.75 | 59.80    | 69.31 |
| QRNet                  | CVPR'22 | Swin | BERT | 72.03 | 65.94    | 56.90 | 40.70 | 13.35 | 60.82    | 75.39 |
| VLTVG                  | CVPR'22 | ResNet-50 | BERT | 71.84 | 66.54    | 57.79 | 41.63 | 14.62 | 60.78    | 70.69 |
| VLTVG                  | CVPR'22 | ResNet-101 | BERT | 73.50 | 68.13    | 59.93 | 43.45 | 15.31 | 62.48    | 73.86 |
| MGVLF                  | TGRS'23 | ResNet-50 | BERT | 72.19 | 66.86    | 58.02 | 42.51 | 15.30 | 61.51    | 71.80 |
| **Ours:**              |  |  |  |  |          |  |  |  |          |  |
| LPVA                   | - | ResNet-50 | BERT | **78.03** | **73.32** | **62.22** | **49.60** | **25.61** | **66.20** | **76.30** |

### Comparison with the SOTA methods for LPVA on the test set of DIOR-RSVG

| Methods | Venue  | Visual Encoder | Language Encoder | Pr@0.5                | Pr@0.6 | Pr@0.7 | Pr@0.8 | Pr@0.9 | meanIoU | cmuIoU |
|---------|--------|----------------|------------------|-----------------------|------|-------|-------|-------|--------|-------|
| **One-stage:** |  |  |  |                       |  |  |  |  |  |  |
| ZSGNet | ICCV'19 | ResNet-50 | BiLSTM | 51.67                 | 48.13 | 42.30 | 32.41 | 10.15 | 44.12 | 51.65 |
| FAOA | ICCV'19 | DarkNet-53 | BERT | 67.21                 | 64.18 | 59.23 | 50.87 | 34.44 | 59.76 | 63.14 |
| ReSC | ECCV'20 | DarkNet-53 | BERT | 72.71                 | 68.92 | 63.01 | 53.70 | 33.37 | 64.24 | 68.10 |
| LBYL-Net | CVPR'21 | DarkNet-53 | BERT | 73.78                 | 69.22 | 65.56 | 47.89 | 15.69 | 65.92 | 76.37 |
| **Transformer-based:** |  |  |  |                       |  |  |  |  |  |  |
| TransVG | CVPR'21 | ResNet-50 | BERT | 72.41                 | 67.38 | 60.05 | 49.10 | 27.84 | 63.56 | 76.27 |
| QRNet | CVPR'22 | Swin | BERT | 75.84                 | 70.82 | 62.27 | 49.63 | 25.69 | 66.80 | 83.02 |
| VLTVG | CVPR'22 | ResNet-50 | BERT | 69.41                 | 65.16 | 58.44 | 46.56 | 24.37 | 59.96 | 71.97 |
| VLTVG | CVPR'22 | ResNet-101 | BERT | 75.79                 |72.22 | 66.33 | 55.17 | 33.11 | 66.32 | 77.85 |
| MGVLF | TGRS'23 | ResNet-50 | BERT | 75.98 | 72.06 | 65.23 | 54.89 | 35.65 | 67.48 | 78.63 |
| **Ours:** |  |  |  |                       |  |  |  |  |  |  |
| LPVA | - | ResNet-50 | BERT | 82.27                 | **77.44** | **72.25** | **60.98** | **39.55** | **72.35** | **85.11** |

## Requirements
- Python 3.6.13
- PyTorch 1.9.0
- NumPy 1.19.2
- cuda 11.1
- opencv 4.5.5
- torchvision

## Citation
If you found this code useful, please cite the paper. Welcome 👍Fork and Star👍, then I will let you know when we update.

```
@ARTICLE{10584552,
  author={Li, Ke and Wang, Di and Xu, Haojie and Zhong, Haodi and Wang, Cong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images}, 
  year={2024},
  volume={62},
  pages={1-13},
  doi={10.1109/TGRS.2024.3423663}}

```
