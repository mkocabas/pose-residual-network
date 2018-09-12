# Pose Residual Network

This repository contains a Keras implementation of the Pose Residual Network (PRN) presented in our ECCV 2018 paper:

Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. [Arxiv](https://arxiv.org/abs/1807.04067)

PRN is described in Section 3.2 of the  paper.


## Getting Started
We have tested our method on [COCO Dataset](http://cocodataset.org)

### Prerequisites

```
python
tensorflow
keras
numpy
tqdm
pycocotools
progress
scikit-image
```

### Installing

1. Clone this repository: 
`git clone https://github.com/mkocabas/pose-residual-network.git`

2. Install [Tensorflow](https://www.tensorflow.org/install/).

3. ```pip install -r src/requirements.txt```

4. To download COCO dataset train2017 and val2017 annotations run: `bash data/coco.sh`. (data size: ~240Mb)

## Training

`python main.py`

For more options take a look at `opt.py`

## Results
Results on COCO val2017 Ground Truth data.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.894
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.971
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.912
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.875
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.918
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.909
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.972
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.928
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.896
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.947
```

## License

## Other Implementations

[Pytorch Version](https://github.com/salihkaragoz/pose-residual-network-pytorch)


## Citation
If you find this code useful for your research, please consider citing our paper:
```
@Inproceedings{kocabas18prn,
  Title          = {Multi{P}ose{N}et: Fast Multi-Person Pose Estimation using Pose Residual Network},
  Author         = {Kocabas, Muhammed and Karagoz, Salih and Akbas, Emre},
  Booktitle      = {European Conference on Computer Vision (ECCV)},
  Year           = {2018}
}
```
