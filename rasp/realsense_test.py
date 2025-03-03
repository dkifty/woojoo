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
