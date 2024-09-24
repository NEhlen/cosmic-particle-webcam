import cv2
import matplotlib.pyplot as plt


cap0 = cv2.VideoCapture(0)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
cap0.set(cv2.CAP_PROP_SHARPNESS, 255)
# cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
#
# cap0.set(cv2.CAP_PROP_EXPOSURE, -10)
# cap0.set(cv2.CAP_PROP_FPS, 20)


cap1 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
cap1.set(cv2.CAP_PROP_SHARPNESS, 255)
# cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
# cap1.set(cv2.CAP_PROP_EXPOSURE, -7)
# cap1.set(cv2.CAP_PROP_FPS, 10)


cap2 = cv2.VideoCapture(4)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
cap2.set(cv2.CAP_PROP_SHARPNESS, 255)
# cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
# cap2.set(cv2.CAP_PROP_EXPOSURE, -7)
# cap2.set(cv2.CAP_PROP_FPS, 10)


while True:
    cap0.grab()
    cap1.grab()
    cap2.grab()
    ret0, frame0 = cap0.retrieve()
    ret1, frame1 = cap1.retrieve()
    ret2, frame2 = cap2.retrieve()
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame0", frame0)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame1", frame1)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame2", frame2)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap0.release()
cap1.release()
cap2.release()
