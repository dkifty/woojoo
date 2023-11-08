1. conda env 를 yaml파일로 export하기
```c
conda activate env_name
conda env export > env_name.yaml
# 가상환경 이름 : env_name

# 복사할 곳에서
conda env create -f env_name.yaml
conda activate env_name
```

2. 같은 환경 내 환경 복사
```c
conda create -n env_name_copy --clone env_name
```

3. 커널 삭제(jupyter)
```c
jupyter kernelspec list # 커널 확인
jupyter kernelspec uninstall env_name
```
