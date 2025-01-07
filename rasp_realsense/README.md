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
