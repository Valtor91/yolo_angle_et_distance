
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import math



model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
whith = 640
heith = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, whith)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, heith)
angle_deg = 0


track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if ret:
        # Run YOLO inference on the frame
        #results = model(frame)
        results = model.track(frame, show=False, persist=True, tracker="tracker.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()


        result=results[0]
        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            cls_list = result.boxes.cls.cpu().tolist()




            frame = result.plot()

            # Plot the tracks
            for box, track_id, cls_id in zip(boxes, track_ids, cls_list):
                x, y, w, h = box
                distance = (2*(470/int(h)))*100

                track = track_history[track_id]

                if float(cls_id) == 0:
                    print()
                    cv2.putText(
                        frame,
                        f"{distance}",
                        (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        2)




                    angle_rad = math.atan((x-320)/ int(distance))
                    angle_deg = math.degrees(angle_rad)

                    cv2.putText(
                        frame,
                        f"Direction: {int(angle_deg)}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)




                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:
                    track.pop(0)


                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)




        cv2.imshow("YOLO Inference", frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()