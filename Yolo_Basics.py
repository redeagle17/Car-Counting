from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-weights/yolov8n.pt') # Here 'yolov8n.py' is the yolo  weights
results = model("Images/bus.jpg", show=True)

cv2.waitKey(0)
