# Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images 
This is the offical PyTorch code for paper "Language-Guided Progressive Attention for Visual Grounding in Remote Sensing Images"  
The code of our method will be open source after the paper is published.  
## OPT-RSVG Dataset 
Download our dataset files. We build the first large-scale dataset for RSVG, termed OPT-RSVG, which can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1e_wOtkruWAB2JXR7aqaMZMrM75IkjqCA?usp=drive_link). 

The download link is available below:
https://drive.google.com/drive/folders/1e_wOtkruWAB2JXR7aqaMZMrM75IkjqCA?usp=drive_link

The dataset contains 25452 RS images and 48952 image-query pairs, with expressions provided in both Chinese and English versions.
![OPT-RSVG Dataset](https://github.com/like413/OPT-RSVG/blob/main/fig/OPT-RSVG.png)
# Ttraining, validation and test samples for OPT-RSVG
| Class Name              | Training | Validation | Test  |
|-------------------------|----------|------------|-------|
| C01: airplane           | 979      | 230        | 1142  |
| C02: ground track field | 1600     | 365        | 2066  |
| C03: tennis court       | 1093     | 284        | 1313  |
| C04: bridge             | 1699     | 452        | 2212  |
| C05: basketball court   | 1036     | 263        | 1385  |
| C06: storage tank       | 1050     | 271        | 1264  |
| C07: ship               | 1084     | 243        | 1241  |
| C08: baseball diamond   | 1477     | 361        | 1744  |
| C09: T junction         | 1663     | 425        | 2055  |
| C10: crossroad          | 1670     | 405        | 2088  |
| C11: parking lot        | 1049     | 268        | 1368  |
| C12: harbor             | 758      | 209        | 953   |
| C13: vehicle            | 3294     | 811        | 4083  |
| C14: swimming pool      | 1128     | 308        | 1563  |
| Total                   | 19580    | 4895       | 24477 |
