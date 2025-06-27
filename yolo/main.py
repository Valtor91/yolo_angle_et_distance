import cv2
from ultralytics import YOLO


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Charger le modèle YOLOv11
model = YOLO("yolo11n")

while True:

    ret, frame = cap.read()



    results = model(frame)


    annotated_frame = results[0].plot()


    cv2.imshow("Camera", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
