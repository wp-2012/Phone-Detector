# Phone-Detector
```python
import time 
import pygame
pygame.mixer.init()
pygame.mixer.music.load("The Longest Day March แปลไทย.mp3") 


import cv2
from ultralytics import YOLO

#load the yolo model
model = YOLO("yolov8x.pt")

# start the camera
camera = cv2.VideoCapture(0)
print("camera started")
# starting the timer
start_time = None

while True:
    ret, frame = camera.read()
    if not ret:
        break
    # Run YOLO without the auto-show feature
    results = model.predict(source=frame)
    found_target = False
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id == 65 or class_id == 67:
            found_target = True
    if found_target:
        if start_time is None:
            start_time = time.time()
            print("start the timer of death")
        else:
            time_passed = time.time() - start_time
            if time_passed > 5:
                print("it's go time")
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
    else:
        start_time = None
        pygame.mixer.music.stop()

    # Draw the boxes on the frame manually
    annotated_frame = results[0].plot()

    # Show the frame in one single, continuous window
    cv2.imshow("Camera Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()
```
