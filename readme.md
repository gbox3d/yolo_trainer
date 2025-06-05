# yolo 통합 트래이너 핼퍼 프로잭트


## 설치 및 실행 방법

```bash
python dataset_split.py --help
기본 사용 (이동 모드)
python dataset_split.py droneAI 

# 복사 모드로 실행
python dataset_split.py droneAI --copy

# 명시적으로 이동 모드 지정
python dataset_split.py droneAI --move

# 복사 모드 + 다른 옵션들
python dataset_split.py droneAI --copy --val-ratio 0.3 --output new_dataset

# 이동 모드 + 다른 옵션들
python dataset_split.py droneAI --val-ratio 0.2 --output datasets --seed 123
```

## detection trainer 실행 방법

```bash
yolo detect train data=dataset.yaml model=yolo11n.yaml epochs=100 imgsz=640
```
