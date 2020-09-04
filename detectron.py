import  detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

import  numpy as np
import os ,json, random
import cv2
import base64

from detectron2 import  model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#img = cv2.imread('elon.jpg')
#cv2.imshow("Elon",img); cv2.waitKey(0); cv2.destroyAllWindows()

cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def filter_objects(outputs,class_id=0):
    person_indices = []
    for i in range(len(outputs['instances'])):
        if (outputs['instances'][i].pred_classes) == 0:
            person_indices.append(i)

    persons = outputs['instances'][person_indices]
    return persons


def predict_mask(img_path):
    img = cv2.imread(img_path)
    outputs = predictor(img)

    persons = filter_objects(outputs,0)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(persons.to("cpu"))

    mask_img = out.get_image()[:, :, ::-1]
    cv2.imwrite("elon_masked.jpg", mask_img)
    _, buffer = cv2.imencode(".jpg", mask_img)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return b64_img




