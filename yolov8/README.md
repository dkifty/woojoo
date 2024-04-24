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
batch = 16
epochs=200
resize_img=1024
iou=0.5
conf=0.5

methods='seg'
model_size='s'
track=False
track_video=False

train=True
save_txt=True
save=True
show_conf=True
show_label=True

# device
device = 0                       # device = 0 or 1 or 2 or 1,2 or cpu
```

```python
from custom_data_preprocessing import make_label_file
make_label_file('Color_checker', 'Flower', 'Fruit_ripen', 'Fruit_unripen', 'Obstacle', 'Old_leaves', 'Picking_point', 'Runner', 'Unidentified')
```

```python
from custom_data_preprocessing import label_name_check
label_name_check(img_format='jpg', label_format='json')
```

```python
from custom_data_preprocessing import data_preprocessing
data_preprocessing(label2coco = label2coco, coco2yolo = coco2yolo, img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO, annotation = annotation, image_size=image_size)
```

```python
from make_yolo_config import make_yolo_config
make_yolo_config()
```

```python
from yolov8_run import yolov8
yolov8(track=track, track_video=track_video, methods=methods, train=train, model_size=model_size, yolo_config='./yolo_configs/data/custom.yaml', imgsz=imgsz, epochs=epochs, batch=batch, device=device, iou=iou, conf=conf, save_txt=save_txt, save=save, show_conf=show_conf, show_label=show_label)
```
