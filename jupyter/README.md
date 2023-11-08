1. config 파일 생성
```c
jupyter lab --generate-config
```

2. config 파일 수정
```c
sudo gedit ~/.jupyter/jupyter_lab_config.py
```

```c
c = get_config()  # noqa
c.JupyterApp.config_file_name = 'jupyter_lab_config.py'
c.ServerApp.allow_origin = '*'
c.ServerApp.notebook_dir = '/home/Desktop/'
c.ServerApp.ip = '192.168.100.12'
	- LAN 일때 : 공인 ip 적기
	- wifi 일때 : 공유기 설정에서 내부 아이피 고정 후 내부아이피 적기
c.ServerApp.open_browser = True
c.ServerApp.port = 8889
c.ServerApp.password = 'argon:어찌구저찌구' # 바로 뒤 설명
```

3. password
```c
ipython
```
```python
>>> from notebook.auth import passwd
>>> passwd()
Enter password:temp (실제 입력시에는 안보인다.)
Verify password:temp (실제 입력시에는 안보인다.)
'argon2:$argon2id$v=19$m=10240,t=10,p=8$5uKI71sd9EQ6SChZMRt1qA$hc7R2Ea+dCoci9uYidjiFN0D5uxG1lrtY3Kyf/G25Rc'
```

고정 ip
- 공유기에 접속 (공인 ip주소 치면 들어가짐)
- 로그인
- 고급설정 - 네트워크관리 - DHCP 서버 설정
- 맨 밑 목록 중 해당하는거 찾아서 위로 등록하기

여기까지 하면 LAN을 연결하는 경우는 접속됨
- 공인IP:8889 이런식으로 들어가면 됨

WIFI쓰는 경우
- 공유기 포트포워딩 해줘야함
- NAT/라우터 관리
- 포트포워딩 설정
    - 내부 IP주소 아까 고정해놨던 주소
    - 프로토콜 TCP
    - 외부포트 8889
    - 내부포트 8889
 
방화벽뚫기
```c
sudo ufw enable
sudo ufw allow 8889
```

접속
- 공인IP:8889
