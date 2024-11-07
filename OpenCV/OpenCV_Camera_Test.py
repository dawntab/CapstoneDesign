import cv2

# GStreamer 파이프라인으로 카메라 열기
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
                       "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                       "videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)

# 카메라 열림 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 프레임을 반복해서 읽음
while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임을 창에 표시
    cv2.imshow('Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 창 닫기
cap.release()
cv2.destroyAllWindows()