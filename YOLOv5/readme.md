## 데이터셋 준비하기

### 데이터셋 생성 유틸 (voc2yolo)

라벨링툴로 작업해서 저장된 포멧은 pascalVoc 포멧이므로 이것을 yolov5로 훈련시키기위해서는 변환이 필요한다.  
yolo형식에 맞는 트레인 세트 구성한다.(train,valid 폴더를 생성하고 데이터를 분리해서 넣는다.)  
.yaml로 괸 설정파일을 지정한다. 여기에는 라벨링정보와 데이터셋 위치등이 기재되어있다.  

config 파일 예시  
```yaml
path: /home/gbox3d/work/datasets/digit/set_7  # dataset root dir
train: train  # train images (relative to 'path') 128 images
val: valid  # val images (relative to 'path') 128 images
test:  test # test images (optional)
voc : voc # voc format
split:  # define train/val set split ratio
  train : 0.8
  val : 0.2

# Classes
nc: 10  # number of classes
names: 
- dight_0
- dight_1
- dight_2
- dight_3
- dight_4
- dight_5
- dight_6
- dight_7
- dight_8
- dight_9
```

사용 예
```sh
# python voc2yolo.py --dataset-path ~/work/dataset/handsign
# python voc2yolo.py --dataset-path ~/work/dataset/madang_v3

python voc2yolo.py --config-file=/home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/madang.yaml
python voc2yolo.py --config-file=/home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml

# 직접 파일 분류해서 넣기 
python voc2yolo_simple.py --ds-path=/home/gbox3d/work/datasets/digit --src=voc_train --dest=train
```

### roboflow 사용하기
비록 모든데이터를 공개해야하지만 객체감지와 새그멘테이션에 대한 라벨링툴을 무료로 제공하고 각종 아규멘테이션기능도 제공한다.  

[로보플로우](https://roboflow.com/)

### makesense ai
로보플로우처럼 웹형태이면서 무료로 제공하는 라벨링툴이다. 뿐만아니라 소스 코드가 공개되어있다.  
서버로데이터를 전송하는 방식이 아닌 로컬에서 처리하는 방식이라 비공개 데티어 처리가 가능하다. 단 기능이 좀 부족하다.  
[메이크센스](https://www.makesense.ai/)  
[소스코드](https://github.com/SkalskiP/make-sense)  

## 학습

yolov5에 저장소에있는 train.py 을 사용하여 훈련한다.  
실행시 PYTHONPATH 환경변수에 yolov5 폴더를 지정해야 한다.  

--data :  데이터셋 폴더 경로
--epochs :  에폭횟수  
--cfg :  설정파일  
--device:  gpu 지정  
--projrct:  결과물 출력 디랙토리 경로  

훈련시키고자 하는 레이어모델의 종류를 선택하려면 cfg,weights 옵션을 사용한다.  
--weights 옵션을 사용하여 원하는 모델의 미리 훈련된 가중치 파일을 지정할 수 있다.  
지정한 가중치 파일이 없으면 자동으로 다운 받는다.  

처음부터 랜덤하게 시작점을 지정하여 훈련하려면 -cfg 옵션을 사용하여 원하는 모델의 cfg 파일을 지정한다.
```
--cfg /home/gbox3d/work/visionApp/yolov5/models/yolov5s.yaml   
```
일반적으로 weights 옵션을 사용한다고한다.  

-hyp 옵션으로 hyper parameter 를 지정할수있다.
기본적인 하이퍼 마라메터 값은 yolov5/data/hyps/hyp.scratch.yaml 에 정의되어 있다. 이것을 가져다 쓰면된다.  

--device 옵션으로 디아비스(gpu)가 여러개일때 사용할 디바이스들을 지정할수있다. 
만약 여러개를 사용하려면 아래와 같이 orch.distributed.run 을 사용하여 실행시킨다.(이것 없이 사용하면 속도가 느려진다.)
--nproc_per_node 옵션으로 사용할 gpu의 갯수를 지정할수있다.  
0,1 디바이스를 사용하여 훈련시키는 예>
```sh
python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1
```

조기 종료 지정 :
--patience 300` or use `--patience 0` to disable EarlyStopping.         


훈련 커멘드 예>
```sh
YOLOV5PATH=/home/gbox3d/work/visionApp/yolov5 
PYTHONPATH=$YOLOV5PATH python $YOLOV5PATH/train.py \
--data /home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml \
--weights yolov5n.pt \
--batch 8 \
--epoch 10 \
--img 640 \
--device 1 \
--save-period 5 \
--hyp /home/gbox3d/work/visionApp/yolov5/data/hyps/hyp.scratch.yaml 
```

### 훈련이어서 하기
훈련중 체크 포인트를 저장하도록 --save-period 옵션을 사용한다.  
500번마다 체크 포인트를 지정하려면 --save-period 500 옵션을 사용한다.  
체크포인트를 지정하지 않으면 훈련재계를 위한 --resume 옵션을 사용할수 없다.
--resume 옵션으로 체크 포인트 파일을 지정하여 훈련을 재개할 수 있다.  
예를 들어 epoch5.pt부터 재계하려면 --resume /경로명/epoch5.pt 옵션을 사용한다.


### 특정 gpu 1개만 지정하여 사용하기


```sh
# gtx-1060 6GB
#bash train_yolov5_single.sh -d ~/work/datasets/digit -e 10000 -m /home/gbox3d/work/visionApp/yolov5/models/yolov5s.yaml  -b 22 -t /home/gbox3d/work/visionApp/yolov5 --datacfg /home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml  -g 1

#PYTHONPATH=/home/gbox3d/work/visionApp/yolov5 python train.py --data /home/gbox3d/work/visionApp/daisy_project/trainer/yolo_v5/config/digit_set_7.yaml --epochs 10000 --batch 22 --cfg /home/gbox3d/work/visionApp/yolov5/models/yolov5s.yaml --device 1

PYTHONPATH=../../yolov5 python ../../yolov5/train.py --data ./config/madang.yaml --epochs 10000 --batch -1 --cfg ../../yolov5/models/yolov5s.yaml --device 0 --project ./output/runs/train/madang --save-period 1000

PYTHONPATH=../../yolov5 python /home/gbox3d/work/visionApp/yolov5/segment/train.py --img 640 --batch 16 --epochs 10000 --data /home/gbox3d/work/datasets/hhgun/data.yaml --weights yolov5m-seg.pt --device 0 --project ./output/runs/train/hhgun --save-period 5000 --patience 500

```

### sehmentation 모델 학습


```sh
PYTHONPATH=../../yolov5 python ../../yolov5/home/segment/train.py --img 640 --batch 16 --epochs 5000 --data /home/gbox3d/work/datasets/hhgun/data.yaml --weights yolov5l-seg.pt --device 0 --project ./output/runs/train/hhgun_l --save-period 500
```

## 텐서보드 사용하기

사용 예
```
cd ~/work/dataset/handsign/workspace/yolov5_train_jobs/
tensorboard --logdir ./
```



