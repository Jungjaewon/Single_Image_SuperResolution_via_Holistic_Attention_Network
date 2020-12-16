# Single_Image_SuperResolution_via_Holistic_Attention_Network

This repository implements the paper ["Single Image Super-Resolution via a Holistic Attention Network"](https://arxiv.org/pdf/2008.08767.pdf) published in ECCV2020.

## Requirements
* python3.6+
* pytorch 1.6.0
* others.

## Usage
training a model
```bash
python3 main.py --config config_patch.yml
```

testing a model
```bash
Not implmented yet
```

## Architecture
![architecture](img/overview.png)
## Results
![shifted_result](img/100-images_shifted.jpg)
![normalresult1](img/100-images.jpg)
![normalresult2](img/300-images.jpg)

## Comments
 In this implementation, the triplet loss function is meaningless. It always show zeros for scaled dot product and l2 norm distance, if I am wrong, please make issue. Without the triplet loss, we can obtain good results. Even if a model is trained only 2 epochs, the model shows meaningful results.
## Reference
1. dataset : https://data.vision.ee.ethz.ch/cvl/DIV2K/