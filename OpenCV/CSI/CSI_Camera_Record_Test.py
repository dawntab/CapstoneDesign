import cv2

# GStreamer 파이프라인으로 카메라 열기
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, "
                       "format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
                       "videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정 (예: 'XVID'는 .avi 파일)
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))  # 파일명, 코덱, FPS, 해상도 설정

# 카메라 열림 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 프레임을 반복해서 읽고 저장
while True:
    ret, frame = cap.read()  # 프레임 읽기

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임을 파일에 저장
    out.write(frame)

    # 프레임을 창에 표시
    cv2.imshow('Camera', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라와 파일 닫기
cap.release()
out.release()
cv2.destroyAllWindows()