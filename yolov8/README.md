```c
import sys
sys.path.append('./utils')
```
```python
# parameters
label2coco = True                # if True - make labelme format annotation to coco format annotation // if False show information of train/valid/test images, annotations for each classes already made
coco2yolo = True          # if True - make coco format annotation to yolo format annotation and make yolo config files // if False just check the config files
img_format = 'jpg'               # default is jpg // you can put other format of image files -> string type
label_format = 'json'            # default is json // you can put other format of annotation files -> string type
change_label_name = False        # you can change the label names in annotation files // format(a,b,c,d is str type) : change_label_name = {a:b, c:d}
split_rate = False               # train/valid/test split rate // default is 0 - 0.9*0.8 / 0.9*0.8 - 0.9 / 0.9 - 1 // format(int type in list) : split_rate = [0.7, 0.2, 0.1]
FOLDERS = ['./data_annotated_train', './data_annotated_valid', './data_annotated_test']                # you can change the name of train/valid/test folder name // but dont do that.... please...
FOLDERS_COCO = ['./data_dataset_coco_train', './data_dataset_coco_valid', './data_dataset_coco_test']  # you can change the name of coco form train/valid/test folder name // but dont do that.... please...
annotation = 'annotations.json'  # default is annotations.json // if annotatino file have other name // annotation = annotations.json (string type)
image_size=(3840,2160)           # if you have other size of image // image_size = (3840, 2160) (default / tuple(int, int))
```

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

from custom_data_preprocessing import make_label_file
make_label_file('Color_checker', 'Flower', 'Fruit_ripen', 'Fruit_unripen', 'Obstacle', 'Old_leaves', 'Picking_point', 'Runner', 'Unidentified')

from custom_data_preprocessing import label_name_check
label_name_check(img_format='jpg', label_format='json')

from custom_data_preprocessing import data_preprocessing

# run
data_preprocessing(label2coco = label2coco, coco2yolo = coco2yolo, img_format=img_format, label_format=label_format, change_label_name=change_label_name, split_rate=split_rate, FOLDERS = FOLDERS, FOLDERS_COCO = FOLDERS_COCO, annotation = annotation, image_size=image_size)

from make_yolo_config import make_yolo_config
make_yolo_config()

from yolov8_run import yolov8
yolov8(track=track, track_video=track_video, methods=methods, train=train, model_size=model_size, yolo_config='./yolo_configs/data/custom.yaml', imgsz=imgsz, epochs=epochs, batch=batch, device=device, iou=iou, conf=conf, save_txt=save_txt, save=save, show_conf=show_conf, show_label=show_label)
