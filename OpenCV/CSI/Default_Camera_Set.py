import cv2

# 우선 GStreamer 파이프라인으로 시도 (주로 CSI 카메라에 적합)
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink", 
    cv2.CAP_GSTREAMER
)

# 카메라 열기에 실패하면 일반 USB 카메라 모드로 전환
if not cap.isOpened():
    print("GStreamer 파이프라인을 사용할 수 없습니다. 일반 카메라 모드로 전환합니다.")
    cap = cv2.VideoCapture(0)  # 일반 모드로 기본 카메라 사용
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

# 카메라 열림 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 프레임 읽기 및 표시 루프
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()