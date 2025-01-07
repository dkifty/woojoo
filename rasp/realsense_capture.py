import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime, timedelta

# 저장 디렉토리 설정
output_dir = "output_images" #########
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
    
    # RealSense 파이프라인 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # 파이프라인 시작
    pipeline.start(config)
    print("Waiting for 2 seconds before capturing the frame...")
    time.sleep(2)  # 초기화 및 안정화를 위해 대기

    # 프레임 가져오기
    frames = pipeline.wait_for_frames(timeout_ms=10000)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire Depth or Color frame.")

    # Depth 이미지 처리
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Color 이미지 처리
    color_image = np.asanyarray(color_frame.get_data())

    # 이미지 저장
    depth_file = os.path.join(output_dir, "{timestamp:%Y-%m-%d-%H%M%S}_depth_image.png")
    color_file = os.path.join(output_dir, "{timestamp:%Y-%m-%d-%H%M%S}_color_image.jpg")

    cv2.imwrite(depth_file, depth_colormap)
    cv2.imwrite(color_file, color_image)
    
    np.save(depth_file.replace(".png", ".npy"), depth_image)
    
    print(f"Saved images to '{output_dir}'.")

finally:
    # 자원 해제
    pipeline.stop()


    print(f"Saved images to '{output_dir}'.")

finally:
    # 자원 해제
    pipeline.stop()
