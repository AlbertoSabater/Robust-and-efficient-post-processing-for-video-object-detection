# Robust and efficient post-processing for Video Object Detection (REPP)

[[Paper](http://webdiis.unizar.es/~anacris/papers/20IROS_Sabater.pdf)]

__REPP__ is a learning based post-processing method to improve video object detections from any object detector. REPP links detection accross frames by evaluating their similarity and refines their classification and location to suppress false positives and recover misdetections.

<p align="center"><img src="./figures/pipeline.png" alt="Post-processing pipeline" width="450"/></p>

REPP improves video detections both for specific Image and Video Object Detectors and it supposes a light computation overhead.

<p align="center"><img src="./figures/results_table.png" alt="Results" width="1000"/></p>


## Installation

REPP has been tested with Python 3.6.

Its dependencies can be found the the _requirements.txt_ file.

```pip install -r requirements.txt```


## Quick usage guide

Video detections must be stored with pickle in the following format:

```
("video_name", {"frame_001": [ det_1, det_2, ..., det_N ],
                "frame_002": [ det_1, det_2, ..., det_M ]},
                ...)
```

If the stored predictions file contains detections for different videos, they must be saved as a stream of tuples with the above format.

And each detection must have the following format:

```
det_1: {'image_id': image_id,     # Same as the used in ILSVRC if applies
        'bbox': [ x_min, y_min, width, height ],
        'scores': scores,         # Vector of class confidence scores
        'bbox_center': (x,y) }    # Relative bounding box center
```

The _bbox_center_ is bounded by 0 and 1 and referes to the center of the detection when the image has been padded vertically or horizontally to fit a square shape. Check [this code](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection/blob/master/demos/Sequence-Level-Semantics-Aggregation/experiments/selsa/selsa_predictions_to_repp.py#L72) for a better insight.

Post-processed detections can be saved both with the COCO or IMDB format.


```
python REPP.py --repp_cfg cfg.json --predictions_file predictions_fil.pckl --store_coco --store_imdb
```

As a REPP configuration file, you can use either _fgfa_repp_cfg.json_ or _yolo_repp_cfg.json_. The first one works better with high performing detectors such as SELSA or FGFA and the second one works better for lower quality detectors. We recommend to set _appearance_matching_ to false in the config file since it requires a non-trivial training of extra models and it's not mandatory for the performance bossting. If needed, the following config parameters can be tunned:

* _min_tubelet_score_ and _min_pred_score_: threshold used to suppress low-scoring detections. Higher values speeds up the post-processing execution.
* _clf_thr_: threshold to suppress low-scoring detections linking. Lower values will lead to more False Positives and higher ones will lead to fewer detections.
* _recoordinate_std_: lower values lead to a more aggressive recoordinating, lower values to a smoother one.


## Demos

In order to reproduce the results of the paper, you can download the predictions of the different models from the following [link]() and locate them in the project folder as structured in the downloaded folder. 

Following commands will apply the REPP post-processing and will evaluate the results by calculating the mean Average Precision for different object motions:

```
# YOLO
python REPP.py --repp_cfg yolo_repp_cfg.json --predictions_file './demos/YOLOv3/predictions/old_preds.pckl' --evaluate --store_coco --store_imdb
# SELSA
python REPP.py --repp_cfg selsa_repp_cfg.json --predictions_file './demos/Sequence-Level-Semantics-Aggregation/predictions/old_preds.pckl' --evaluate --store_coco --store_imdb
# FGFA
python REPP.py --repp_cfg fgfa_repp_cfg.json --predictions_file './demos/Flow-Guided-Feature-Aggregation/predictions/old_preds.pckl' --evaluate --store_coco --store_imdb
```

Instead of downloaded the base predictions, you can also compute them. To do so, you must install the proper dependencies for each model as specified in the original model repositories ([YOLOv3](https://github.com/AlbertoSabater/Robust-and-efficient-post-processing-for-video-object-detection/tree/master/demos/YOLOv3), [FGFA](https://github.com/guanfuchen/Flow-Guided-Feature-Aggregation), [SELSA](https://github.com/happywu/Sequence-Level-Semantics-Aggregation)). Then execute the following commands:

```
# YOLO

# SELSA
cd demos/Sequence-Level-Semantics-Aggregation/
python experiments/selsa/get_repp_predictions.py --dataset_path 'path_to_dataset/ILSVRC2015/'
# FGFA
cd demos/Flow-Guided-Feature-Aggregation/fgfa_rfcn/
python get_repp_predictions.py  --det_path 'path_to_dataset/ILSVRC2015/'


```

Descargar modelos...


YOLO predictions from video


Please cite:
