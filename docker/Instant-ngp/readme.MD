instant ngp - 도커 사용 
ubuntu 22.04 원격 서버 기준 

(1) COLMAP DB 생성 + feature 추출
colmap feature_extractor \
  --database_path database.db \
  --image_path ./images \
  --SiftExtraction.use_gpu 0

(2) feature 매칭
colmap exhaustive_matcher \
  --database_path database.db \
  --SiftMatching.use_gpu 0

# (3) sparse reconstruction 수행
mkdir sparse
colmap mapper \
  --database_path database.db \
  --image_path ./images \
  --output_path ./sparse

결과:
sparse/0/cameras.bin, images.bin, points3D.bin 파일 생성됨

colmap model_converter \
  --input_path sparse/0 \
  --output_path sparse/0 \
  --output_type TXT

cd instant-ngp/scripts

python colmap2nerf.py --images ../../lettuce_test/01/images --text ../../lettuce_test/01/sparse/0

docker build -t instant-ngp-cuda118 .

sudo docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  -v "$PWD":/opt \
  -w /opt/instant-ngp \
  instant-ngp-cuda118

cmake . -B build
cmake --build build --config RelWithDebInfo -j $(nproc)

실행
./build/instant-ngp --scene /opt/instant-ngp/scripts/transforms.json --no-gui
