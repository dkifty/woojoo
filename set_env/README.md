# Tensorflow
```c
conda create -n tf27 python=3.9 -y
conda activate tf27

pip install tensorflow-gpu==2.8

pip install ipykernel
python -m ipykernel install --user --name tf28 --display-name tf28

pip install pandas matplotlib scikit-learn scipy opencv-python tqdm
```

```python
import pandas
import matplotlib
import sklearn
import scipy
import cv2
import numpy

print(pandas.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
print(scipy.__version__)
print(cv2.__version__)
print(numpy.__version__)

import tensorflow
tensorflow.__version__

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

# pytorch
```c
conda create -n torch python=3.9 -y
conda activate torch

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install ipykernel
python -m ipykernel install --user --name torch --display-name torch

pip install pandas matplotlib seaborn scipy scikit-learn opencv-python
```

```python
import pandas
import matplotlib
import sklearn
import scipy
import cv2
import numpy

print(pandas.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
print(scipy.__version__)
print(cv2.__version__)
print(numpy.__version__)

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
```

# detection
```c
conda create -n detection python=3.9 -y
conda activate detection

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install ultralytics
pip install tensorflow-gpu==2.11.0

pip install ipykernel
python -m ipykernel install --user --name detection --display-name detection

pip install pandas matplotlib seaborn scipy scikit-learn opencv-python labelme
```

# YOLO v8 counting
```c
conda create -n counting python=3.9 -y
conda activate counting

pip install ultralytics
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
pip3 install -q -r requirements.txt
python3 setup.py -q develop
pip install -q cython_bbox
pip install -q onemetric
pip install -q loguru lap thop

pip uninstall protobuf
pip install protobuf==3.19.0

pip uninstall numpy
pip install numpy==1.23.5

pip install ipykernel
python -m ipykernel install --user --name counting --display-name counting

pip install pandas matplotlib seaborn scipy scikit-learn opencv-python labelme
```
