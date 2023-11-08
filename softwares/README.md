# chrome
```c
sudo apt update && $ sudo apt upgrade
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

# anydesk
```c
sudo apt update
sudo apt -y upgrade
sudo apt --fix-broken install
sudo dpkg -i anydesk_6.2.1-1_amd64.deb
```

# putty
```c
sudo apt update

# ssh 서버 설정
sudo apt-get install -y openssh-server

# ssh 서비스 시작
sudo service ssh start

# ssh 방화벽 허용 설정
sudo ufw enable
sudo ufw allow 22 # default ssh port : 22

# ssh 아이피확인
hostname -I

sudo reboot
```

# 한글세팅
https://shanepark.tistory.com/231 참고

# 랜카드
```c
git clone https://github.com/cilynx/rtl88x2bu.git
cd rtl88x2bu
VER=$(sed -n 's/\PACKAGE_VERSION="\(.*\)"/\1/p' dkms.conf)
sudo rsync -rvhP ./ /usr/src/rtl88x2bu-${VER}
sudo dkms add -m rtl88x2bu -v ${VER}
sudo dkms build -m rtl88x2bu -v ${VER}
sudo dkms install -m rtl88x2bu -v ${VER}
sudo modprobe 88x2bu
```
