import cv2

# 해상도와 프레임 설정
frame_width = 1280  # 가로 해상도
frame_height = 720  # 세로 해상도
fps = 30            # 프레임 속도

# GStreamer 비활성화를 위해 일반 OpenCV 설정 사용 (기본 카메라 열기)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 해상도 및 프레임 속도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, fps)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임을 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Grayscale Webcam", gray_frame)

    # 프레임을 창에 표시
    cv2.imshow("Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()