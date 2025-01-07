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
git checkout development

# 빌드 및 설치
mkdir build && cd build
cmake ../ -DFORCE_LIBUVC=ON -DCMAKE_BUILD_TYPE=Release
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
sudo make install

# 설치확인
realsense-viewer
```
기타 설치
```c
pip3 install opencv-python --break-system-packages
```

# pyrealsense2
pyrealsense2 작동 확인
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
pyrealsense python파일로 rgb, depth 이미지 촬영 후 저장
```python
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 저장 디렉토리 설정
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # RealSense 장치 연결
    ctx = rs.context()
    if len(ctx.devices) == 0:
        raise RuntimeError("No RealSense devices connected.")
    
    # 장치 가져오기
    device = ctx.query_devices()[0]
    sensor = device.first_depth_sensor()
    
    # 파이프라인 없이 단일 프레임 캡처
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 단일 프레임 가져오기
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Depth와 Color 프레임 유효성 확인
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire Depth or Color frame.")

    # Depth 이미지 처리
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Color 이미지 처리
    color_image = np.asanyarray(color_frame.get_data())

    # 이미지 저장
    depth_file = os.path.join(output_dir, "depth_image.png")
    color_file = os.path.join(output_dir, "color_image.jpg")

    cv2.imwrite(depth_file, depth_colormap) # Depth 컬러맵 저장
    np.save(depth_file.replace(".png", ".npy"), depth_image) # Depth 원본 데이터 저장
    cv2.imwrite(color_file, color_image) # Color 이미지 저장

    print(f"Saved images to '{output_dir}'.")

finally:
    # 자원 해제
    pipeline.stop()
```
