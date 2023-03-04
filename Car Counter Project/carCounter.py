from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

cap = cv2.VideoCapture("../Videos/cars.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

model = YOLO("../Yolo-weights/yolov8n.pt")
mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
lines_coordinates = [400, 297, 673, 297]  # lines_coordinates[0] is one point and lines_coordinates[2] is another point
# lines_coordinates[1] is height from origin and lines_coordinates[2] is height from origin
totalCount = 0
visited_id_list = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(mask, img)  # Will give the required region for counting
    results = model(imgRegion, stream=True)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # To draw bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # Gives coordinates to draw bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)

            # Annotation is required for the detected object
            confidence_level = math.ceil((box.conf[0] * 100)) / 100  # rounding of to 2 decimal places

            # To get class name
            cls = int(box.cls[0])  # box.cls[0] will give us the class id(float so we need to typecast it) and with
            # the help of the id will get the class name from classNames list
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "motorbike" or currentClass == "truck" or \
                    currentClass == "bus" and confidence_level > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {confidence_level}', (max(0, x1), max(35, y1)), scale=0.6,
                #                    thickness=2, offset=3)
                # cvzone.cornerRect(img, bbox, l=9)
                # cornerRect is the  built-in function in cvzone to draw bounding box

                currentArray = np.array([x1, y1, x2, y2, confidence_level])  # Check update function in sort.py
                detections = np.vstack((detections, currentArray))
    resultTracker = tracker.update(detections)
    cv2.line(img, (lines_coordinates[0], lines_coordinates[1]), (lines_coordinates[2], lines_coordinates[3]),
             (0, 0, 255), thickness=2)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        # print(result)
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2.3,
                           thickness=2, offset=3)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=3, rt=2, colorR=(255, 0, 255))

        # Now, we want the center point of the detected object and if it touches the line then increase the count
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        if lines_coordinates[0] < cx < lines_coordinates[2] and lines_coordinates[1] - 10 < cy < lines_coordinates[
            1] + 10:
            if id not in visited_id_list:  # To count the number only once
                totalCount = totalCount + 1
                visited_id_list.append(id)
                cv2.line(img, (lines_coordinates[0], lines_coordinates[1]),
                         (lines_coordinates[2], lines_coordinates[3]),
                         (0, 255, 0), thickness=2)
    # cvzone.putTextRect(img, f'Count {totalCount}', (40, 40), colorT=(255, 255, 255), colorR=(0, 0, 0))
    cv2.putText(img, str(totalCount), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("Region", imgRegion)
    cv2.waitKey(1)
