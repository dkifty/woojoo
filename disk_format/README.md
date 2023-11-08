1. 윈도우 USB 만들기
2. USB 우선부팅
3. 윈도우 설치 첫페이지에서 Shift + f10

```c
diskpart
list disk
sel disk 0 # 지우고자하는 디스크 선택
detail disk
clean
```
4. 재부팅 -> 윈도우 USB 제거 -> ubuntu USB 넣기
