import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/My Video.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits0=[400,620,900,625]
limits1 = [780, 420, 930, 425]

limits2=[1030,620,1550,618]
limits3 = [990, 420, 1140, 420]

totalCount0 = []
totalCount1 = []


while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections0 = np.empty((0, 5))
    detections1 = np.empty((0, 5))



    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass0 = classNames[cls]
            currentClass1 = classNames[cls]


            if currentClass0 == "car" or currentClass0 == "truck" or currentClass0 == "bus" \
                    or currentClass0 == "motorbike" and conf > 0.3:

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections0 = np.vstack((detections0, currentArray))
                detections1 = np.vstack((detections1, currentArray))
                # print(detections0)
    resultsTracker0 = tracker.update(detections0)
    resultsTracker1 = tracker.update(detections1)

    cv2.line(img, (limits0[0], limits0[1]), (limits0[2], limits0[3]), (0, 0, 255), 3)
    cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 3)

    cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 3)
    cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 0, 255), 3)
    for result in resultsTracker0:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits1[0] < cx < limits1[2] and limits1[1] - 15 < cy < limits1[1] + 15:
            if totalCount0.count(id) == 0:
                totalCount0.append(id)
                cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)



    # right side
    for result in resultsTracker1:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits2[0] < cx < limits2[2] and limits2[1] - 15 < cy < limits2[1] + 15:
            if totalCount1.count(id) == 0:
                totalCount1.append(id)
                cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 3)







    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str("Total In-"),(50,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.putText(img,str(len(totalCount0)),(430,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.putText(img,str("Total Out-"),(800,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.putText(img,str(len(totalCount1)),(1250,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
