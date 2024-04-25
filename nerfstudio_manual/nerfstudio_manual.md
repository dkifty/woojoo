# 0. 폴더에 비디오 파일 넣기
# 그 폴더에서 실행

# 1. set env
```c
conda create -n nerfstudio python=3.11.7
conda activate nerf

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio
```

# 2. colmap
```c
ns-process-data video --data GX010824.MP4 --output-dir ./ --matching-method exhaustive --sfm-tool colmap --num-frames-target 321 --num-downscales 0

## parameters
--data : 동영상 데이터 입력
--output-dir : colmap 결과 어디에 저장할지
--matching-method : exhaustive로 (sequential 성능이 너무 안좋음)
--sfm-tool : colmap or hloc
--num-downscales 0 : 원래 이미지에 대해서만 / e.g. 3(default) downscale을 1/2 1/4 1/8해서 영상->이미지로 추출
--num-frames-target 300 : 비디오에서 300개 이미지 뺀다는거임(약 300개로 비슷하게 비례해서 추출해줌)

## colmap method hloc 사용시
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .

## 다른 파라미터 수정 원할 시
ns-process-data --help
```

# 3. NERF
```c
ns-train nerfacto --data ./

## nerfacto 대신...
'depth-nerfacto', 'dnerf', 'generfacto', 'instant-ngp', 'instant-ngp-bounded', 'mipnerf', 'nerfacto', 'nerfacto-big', 'nerfacto-huge', 'neus', 'neus-facto', 'phototourism', 'semantic-nerfw', 'splatfacto', 'tensorf', 'vanilla-nerf', 'igs2gs', 'in2n', 'in2n-small', 'in2n-tiny', 'kplanes', 'kplanes-dynamic', 'lerf', 'lerf-big', 'lerf-lite', 'nerfplayer-nerfacto', 'nerfplayer-ngp', 'pynerf', 'pynerf-occupancy-grid', 'pynerf-synthetic', 'seathru-nerf', 'seathru-nerf-lite', 'tetra-nerf', 'tetra-nerf-original', 'volinga', 'zipnerf' 가능
splatfacto 시 
pip install gsplat==0.1.6
```

```python
## http://0.0.0.0:7007/ 
## 학습 끝나면 적당한 크기로 crop
## 1000000 포인트(default) 로 point cloud extract --> generate command에서 나오는 명령어 복사 / point num 은 수정 가능 -> 약 20000000이 적당
## command 에서 ctrl+c 후 위 명령어 붙여넣기
```

# instant-ngp로 colmap
```c
conda create -n instant-ngp python=3.11
git clone --recursive https://github.com/nvlabs/instant-ngp

cd instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j

pip install commentjson commentjson numpy opencv-python-headless pybind11 pyquaternion pyquaternion tqdm imageio==2.9.0 scipy
                     
sudo apt install ffmpeg
sudo apt-get install colmap

mkdir data/garlic ##'garlic'
cd data/garlic

python ../../scripts/colmap2nerf.py --video_in GX010824.MP4 --video_fps 3 --run_colmap --colmap_matcher exhaustive --aabb_scale 16 --overwrite
```
```
