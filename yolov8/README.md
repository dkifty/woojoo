- data_annotated폴더 생성 후 이미지, 라벨 파일 넣기
- git clone으로 utils파일 가져오기

```c
import sys
sys.path.append('./utils')
```

```python
# parameters
label2coco = True                # labelme형식으로 된 데이터셋을 coco형식으로 바꿀것인지(True/False)
coco2yolo = True                 # coco형식으로 된 데이터셋을 yolo형식으로 바꿀것인지(True/False)
img_format = 'jpg'               # 가진 이미지 파일의 포멧
label_format = 'json'            # 가진 annotation 파일의 포멧
change_label_name = False        # 라벨링한 데이터 중 이름 한번에 바꾸고싶을때 // e.g. change_label_name = {a:b, c:d} -> a와 c로 라벨링 된거 전부 b,d로 각각 바뀜
split_rate = False               # 학습/검증/테스트 데이터세트 비율 // default is 0 - 0.9*0.8, 0.9*0.8, 0.9 / 0.9  // e.g. split_rate = [0.7, 0.2, 0.1]
FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test']                # 혹시나 정말 혹시나 폴더 명이 맘에 안들면 바꿀수있음... // 안하는거 추천(어디서 에러뜰지 모름..)
FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']  # 혹시나 정말 혹시나 폴더 명이 맘에 안들면 바꿀수있음... // 안하는거 추천(어디서 에러뜰지 모름..)
annotation = 'annotations.json'  # annotation파일 명 바꾸고싶을때... // annotation = annotations.json (string type) // 안하는거 추천
image_size=(3840,2160)           # 가진 이미지 크기 // image_size = (3840, 2160) (default / tuple(int, int))

# model
batch = 16                       # 배치사이즈
epochs=200                       # 에폭
resize_img=1024                  # 학습할때 이미지 크기 resize해서 함 -> 그 크기
iou=0.5                          # IOU
conf=0.5                         # Confidence score

methods='seg'                    # detection할건지 segmentation 할건지 // e.g. methods='seg' or 'det'
model_size='s'                   # 사용할 yolov8 알고리즘 크기 // e.g. model_size='s' or 'n' or 'm' or 'l' or 'x'
track=False                      # tracking 할건지(각 객체에 id부여) // 이 경우 테스트데이터셋이 아닌 동영상이 input되게 해놓음 // track_video 파라미터도 설정 필요
track_video=False                # tracking 할 동영상 파일 경로 // e.g. track_video='./GX010350.MP4'

train=True                       # 학습 할건지 // 학습 했는데 또 할 필요는 없으니까... // False시 학습했던 weight불러와서 val, test 수행
save_txt=True                    # detecting한 예측 값들을 txt로 저장하는지
save=True                        # default
show_conf=True                   # detecting 시각화 할 때 conf score와 label을 표시할지 아닐지(아래 동일)
show_label=True

# device
device = 0                       # device = 0 or 1 or 2 or '1,2' or 'cpu'
```

```python
from custom_data_preprocessing import make_label_file
make_label_file('Color_checker', 'Flower', 'Fruit_ripen', 'Fruit_unripen', 'Obstacle', 'Old_leaves', 'Picking_point', 'Runner', 'Unidentified')
# 굳이 오름차순으로 안해도 되게 해놓기는 함... labels.txt파일 생성하는 코드
```

```python
from custom_data_preprocessing import label_name_check
label_name_check(img_format='jpg', label_format='json')
# labels.txt 파일에 지정한 라벨명이랑 라벨링 수행한 파일들에서의 라벨명이랑 동일한지 체크
# 라벨링 하다가 오타가 나거나 잘못한것들 찾기 위함
```

```python
from custom_data_preprocessing import data_preprocessing
data_preprocessing(label2coco = label2coco, coco2yolo = coco2yolo, img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO, annotation = annotation, image_size=image_size)
# 코딩해논 작업 순서
## 1. 이미지와 라벨 숫자 맞는지 체크
## 2. change_label_name 지정 시 전부 바꿈 -> 재실행 필요
## 3. labels.txt 파일에 지정한 라벨명이랑 라벨링 수행한 파일들에서의 라벨명이랑 동일한지 체크
## 4. 정해진 비율에 따라 학습, 검증, 테스트 데이터셋 나눔(랜덤하게)
## 5. 각 데이터셋을 coco form으로 바꿈
## 6. 각 데이터셋에 포함된 각 객체의 수 세기
## 7. 각 데이터셋을 yolo form으로 바꿈(labels_det, labels_seg폴더에 들어감)
### 7.1. 이거 때문에 이렇게 만든 데이터로 일반적인 yolo는 안돌아감 돌리고 싶으면 labels_det 또는 labels_seg폴더명을 labels로 바꿔야함
### 7.2. 근데 왜 이렇게 했냐면... detection하고 segmentation 둘다 하는데 할때마다 바꾸기 귀찮아서 yolo내부 코드를 바꿈.. 아래 코드에 모두 포함시켜놔서 따로 할건 없음
```

```python
from make_yolo_config import make_yolo_config
make_yolo_config()
# 학습시 필요한 config파일 생성
```

```python
from yolov8_run import yolov8
yolov8(track=track, track_video=track_video, methods=methods, train=train, model_size=model_size, yolo_config='./yolo_configs/data/custom.yaml', imgsz=imgsz, epochs=epochs, batch=batch, device=device, iou=iou, conf=conf, save_txt=save_txt, save=save, show_conf=show_conf, show_label=show_label)
# 위의 파라미터만 잘 맞춰서 하면 yolo v8 돌아감
```
- 결과는 생성된 results폴더에 저장하게 해놈
```c
## 참고 - 예측 라벨 색상 바꾸는 법
anaconda - envs - 가상환경명 - lib - python3.11(버전에 맞는거) - site-packages - ultralytics - utils - plotting.py 들어가기
class Colors에서 def __init__에서 hex안의 인자들 바꾸면 됨~
```
