# tested
ubuntu 20.04 LTS
nvidia driver 495
cuda 11.3
cuDNN 8.2.1
GPU - RTX A6000

# setting
 우선
```c
sudo apt update && sudo apt upgrade && sudo apt dist-upgrade -y

#기존 nvidia, cuda 완전 삭제
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

설치 가능한 드라이버 확인
```c
ubuntu-drivers devices # 그래픽카드 및 설치 가능한 드라이버를 확인

sudo apt install nvidia-driver-495 # 495 위치에 원하는 버전 기입

sudo apt-get install dkms nvidia-modprobe # nvidia kernel load를 도와주는 modprobe패키지 설치

sudo apt update & upgrade
sudo reboot
nvidia-smi # 재부팅 후 설치 잘 되었는지 확인
```

cuda 설치(11.3)
- https://developer.nvidia.com/cuda-toolkit-archive
버전 선택 - linux - x86_64 - ubuntu - 20.04(사용하는 우분투 버전) - runfile(local) 다운로드
```c
# 커널에서 바로 다운 wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run

sudo sh cuda_11.3.1_465.19.01_linux.run

continue

accept

driver 선택 해제 후 install
```
설치 후 설치 잘되면 summary~~~ 어찌구 Logfile is /var/log/cuda-installer.log 이런식으로 뜨면 성공

cuda version check
```c
nvcc -V
```

cuda toolkit 안뜰때
```c
sudo gedit ~/.bashrc

# bashrc 맨 아래 추가
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# 적용
source ~/.bashrc
```

cuDNN 설치(8.2.1)
- https://developer.nvidia.com/rdp/cudnn-archive
원하는 버전 선택 후 다운

실행
```c
tar xvzf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp cuda/include/cudnn* /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.2.1 /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.2.1  /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.2.1  /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.2.1  /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.2.1  /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.2.1 /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
sudo ln -sf /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn.so.8.2.1 /usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn.so.8

sudo ldconfig # 새로 추가된 라이브러리를 시스템에서 찾을 수 있도록 하기

ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn # 설정 잘 되었는지 확인
## 출력 : libcudnn_ops_train.so.8 → libcudnn ~~~~

cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 # 설치된 cuDNN 버전 확인
```
