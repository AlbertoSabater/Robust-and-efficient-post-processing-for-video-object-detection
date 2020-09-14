# Sequence Level Semantics Aggregation for Video Object Detection

## Introduction
This is an official MXNet implementation of 
[*Sequence Level Semantics Aggregation for Video Object Detection*](https://arxiv.org/abs/1907.06390). (ICCV 2019, oral).
SELSA aggregates full-sequence level information of videos while keeping a simple and clean pipeline. It achieves **82.69**
mAP with ResNet-101 on ImageNet VID validation set.

## Citation
If you use the code or models in your research, please cite with:
```
@article{wu2019selsa,
  title={Sequence Level Semantics Aggregation for Video Object Detection},
  author={Wu, Haiping and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={ICCV 2019},
  year={2019}
}
```

## Main Results


|                                 | <sub>training data</sub>  | <sub>testing data</sub> | <sub>mAP(%)</sub> | <sub>mAP(%)</br>(slow)</sub>  | <sub>mAP(%)</br>(medium)</sub> | <sub>mAP(%)</br>(fast)</sub> |
|---------------------------------|-------------------|--------------|---------|---------|--------|--------|
| <sub>Single-frame baseline</br>(Faster R-CNN, ResNet-101)</sub>   | <sub>ImageNet DET train</br> + VID train</sub> | <sub>ImageNet VID validation</sub> | 73.6 | 82.1 | 71.0 | 52.5 |
| <sub>SELSA</br>(Faster R-CNN, ResNet-101)</sub>  | <sub>ImageNet DET train</br> + VID train</sub> | <sub>ImageNet VID validation</sub> | 80.3| 86.9 | 78.9 | 61.4 |
| <sub>SELSA</br>(Faster R-CNN, ResNet-101, Data Aug)</sub>  | <sub>ImageNet DET train</br> + VID train</sub> | <sub>ImageNet VID validation</sub> | 82.7 | 88.0 | 81.4 | 67.1 |



## Installation

Please note that this repo is based on Python 2.

1. Clone the repository.
~~~
git clone https://github.com/happywu/Sequence-Level-Semantics-Aggregation
~~~

2. Install MXNet following https://mxnet.incubator.apache.org/get_started. We tested our code on MXNet v1.3.0.

3. Install packages via 
~~~
pip install -r requirements.txt
sh init.sh
~~~

## Preparation for Training & Testing

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset, and make sure it looks like this:

	```
	./data/ILSVRC2015/
	./data/ILSVRC2015/Annotations/DET
	./data/ILSVRC2015/Annotations/VID
	./data/ILSVRC2015/Data/DET
	./data/ILSVRC2015/Data/VID
	./data/ILSVRC2015/ImageSets
	```

2. Please download ImageNet pre-trained [ResNet-v1-101](https://1dv.alarge.space/resnet_v1_101-0000.params) model and 
our pretrained [SELSA ResNet-101](https://1dv.alarge.space/selsa_rcnn_vid-0000.params) model manually, and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	./model/pretrained_model/selsa_rcnn_vid-0000.params
	```
## Testing
1. To test the provided pretrained model, run the following command.
    ```
    python experiments/selsa/test.py --cfg experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml --test-pretrained ./model/pretrained_model/selsa_rcnn_vid
    ```
   
You should get the results as reported before.
## Training

3. To train, use the following command
    ```
    python experiments/selsa/train_end2end.py --cfg experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml
    ```
	A cache folder would be created automatically to save the model and the log under `output/selsa_rcnn/imagenet_vid/`.
	
2. To test your trained model
    ```
    python experiments/selsa/test.py --cfg experiments/selsa/cfgs/resnet_v1_101_rcnn_selsa_aug.yaml
    ```
	
## Acknowledge
This repo is modified from [*Flow-Guided-Feature-Aggregation*](https://github.com/msracver/Flow-Guided-Feature-Aggregation).

