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
