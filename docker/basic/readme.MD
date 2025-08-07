도커 기본 명령어

Docker 설치
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

Docker Compose (필요 시)
sudo apt install docker-compose

현재 사용자에 Docker 권한 부여
sudo usermod -aG docker $USER
newgrp docker

NVIDIA GPU를 사용하는 경우 (선택)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker

실행중 컨테이너 조회
sudo docker ps -a

컨테이너 끄기
sudo docker stop 컨테이너이름
sudo docker rm -f 컨테이너이름

모든 컨테이너 중지
sudo docker stop $(sudo docker ps -q)
모든 종료된(Exited) 컨테이너만 삭제
sudo docker container prune
모든 컨테이너 강제 삭제 (실행 중이든 종료 상태든 전부)
sudo docker rm -f $(sudo docker ps -aq)

빌드된 도커 조회
sudo docker images

지우기
sudo docker rmi 이름
