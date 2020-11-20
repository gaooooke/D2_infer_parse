import torch
import numpy as np
import cv2
torch.backends.cudnn.benchmark = True

import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import custom_data
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from imutils.video import VideoStream
import time
import multiprocessing as mp

def Predictor(yaml_file,weight_file,confidence_thr,num_classes):
    """
    Input:
         yaml_file: str. backbone
         weight_file : str. model weight
         confidence_thr: Num. confidence threshold
         num_classes: Num. classes number
    Output:
         return : predictor
    """
    MetadataCatalog.get("custom")
    cfg = get_cfg()
    cfg.merge_from_file(yaml_file)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thr
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    return predictor

yaml_file = "/home/gaokechen/detectron2-master/configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"
weight_file1 = "model_final_het.pth"
confidence_thr = 0.7
num_classes1 = 2
weight_file2 = "model_final.pth"
num_classes2 = 5
predictor = Predictor(yaml_file,weight_file2,confidence_thr,num_classes2)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    ######################################################
    ####          TEST PASS
    #####################################################
    custom_metadata = MetadataCatalog.get("custom")
    print("------------------Start--------------------")
    data_f = '/media/ps/D/TeamWork/chengk/PYTORCH/custom_d2/pic/001526.jpg'
    im = cv2.imread(data_f)
    outputs = predictor(im)

    print(f"detect {len(outputs)} object")
    print("outputs:",outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=custom_metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE  # COCO : ColorMode.IMAGE_BW
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)

