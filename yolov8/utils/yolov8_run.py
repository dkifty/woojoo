#!/usr/bin/env python

from ultralytics import YOLO
import os
import site

def yolov8(track_video=False, track=False, methods='seg', train=True, model_size='s', yolo_config='./yolo_configs/data/custom.yaml', imgsz=1024, epochs=200, batch=16, device=0, iou=0.5, conf=0.5, save_txt=True, save=True, show_conf=True, show_label=True):
    ultralytics_package_utils = os.path.join(site.getsitepackages()[0], 'ultralytics/data/utils.py')
    with open(ultralytics_package_utils, 'r') as f:
        lines = f.readlines()
    lines[33] = "    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels_"+methods+"{os.sep}'  # /images/, /labels/ substrings\n"
    with open(ultralytics_package_utils, 'w') as a:
        for line in lines:
            a.write(line)
    
    if methods == 'seg':
        model_name = 'yolov8'+model_size+'-'+methods
    elif methods == 'det':
        model_name = 'yolov8'+model_size
    else:
        pass
    
    if train==True:
        model = YOLO(model_name)
        if not os.path.exists('results'):
            os.mkdir('results')
        results = model.train(model=model_name, data=yolo_config, imgsz=imgsz, epochs=epochs, batch=batch, device=device, project=os.path.join('results', model_name), exist_ok=True)
        model = YOLO(os.path.join(str(results.save_dir), 'weights/best.pt'))
    elif train==False:
        model = YOLO(os.path.join(os.path.join('results', model_name, 'train/weights/best.pt')))
    else:
        pass
    
    metrics = model.val(split='test', iou=iou, project=os.path.join('results', model_name), exist_ok=True)
    if track==True:
        for r in model.track(source=track_video, conf=conf, save_txt=save_txt, save=save, stream=True, project=os.path.join('results', model_name, track_video.split('/')[-1].split('.')[0]), show=True):
            pass
    elif track==False:
        for r in model.predict(source='data_dataset_coco_test/images', conf=conf, save_txt=save_txt, save=save, stream=True, project=os.path.join('results', model_name), exist_ok=True):
            pass
