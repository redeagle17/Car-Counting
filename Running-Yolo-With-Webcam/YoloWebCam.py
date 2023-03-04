from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # For webcam
# cap.set(3, 1240)
# cap.set(4, 720)

cap = cv2.VideoCapture("../Videos/motorbikes.mp4")

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

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # To draw bounding box
            x1, y1, x2, y2 = box.xyxy[0]  # Gives coordinates to draw bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            cvzone.cornerRect(img, bbox, l=30, t=3, rt=2, colorR=(255, 0, 0), colorC=(0, 255, 255))
            # cornerRect is the  built-in function in cvzone to draw bounding box

            # Annotation is required for the detected object
            confidence_level = math.ceil((box.conf[0] * 100)) / 100  # rounding of to 2 decimal places

            # To get class name
            cls = int(box.cls[0])  # box.cls[0] will give us the class id(float so we need to typecast it) and with
            # the help of the id will get the class name from classNames list
            cvzone.putTextRect(img, f'{classNames[cls]} {confidence_level}', (max(0, x1), max(35, y1)), scale=1,
                               thickness=2, colorT=(0, 0, 0), colorR=(100, 0, 100))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
