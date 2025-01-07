# env
raspberry 4B
os : bookworm (debian 12)
realsense d435i

# librealsense 설치
Intel RealSense SDK(librealsense)를 설치
```c
sudo apt install git cmake build-essential libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev

# librealsense 소스코드 클론
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

# 필요한 스크립트 실행
./scripts/setup_udev_rules.sh
./scripts/patch-realsense-ubuntu-lts.sh

# 빌드 및 설치
mkdir build && cd build
cmake ../ -DFORCE_LIBUVC=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# 설치확인
realsense-viewer
```

# pyrealsense2
```c
pip install pyrealsense2
```

```python
import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 스트리밍 시작
pipeline.start(config)

try:
    while True:
        # 프레임 수집
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # depth 이미지를 가시화
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # RGB와 Depth를 나란히 표시
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()
```
