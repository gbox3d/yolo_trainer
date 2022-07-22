#!/bin/bash

YOLOV5PATH=/home/gbox3d/work/visionApp/yolov5 


PYTHONPATH=$YOLOV5PATH python $YOLOV5PATH/train.py \
--data ../config/madang.yaml \
--weights ../yolov5s.pt \
--project ../runs/train/madang \
--epochs 10000 \
--batch 50 \
--device 0 \
--save-period 1000