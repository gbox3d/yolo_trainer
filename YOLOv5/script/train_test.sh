#!/bin/bash


PYTHONPATH=/home/gbox3d/work/visionApp/yolov5 python3 train.py \
--data /home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml \
--weights yolov5n.pt \
--batch 8 \
--epoch 10 \
--img 640 \
--device 1 \
--save-period 5 \
--hyp /home/gbox3d/work/visionApp/yolov5/data/hyps/hyp.scratch.yaml \
# --resume /home/gbox3d/work/datasets/digit/workspace/yolov5_train_jobs/exp4/weights/epoch5.pt
# --project  \
# --name exp \
#--weights "" \
#--cfg /home/gbox3d/work/visionApp/yolov5/models/yolov5s.yaml \