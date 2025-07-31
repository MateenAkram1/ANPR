import cv2
from ultralytics import YOLO
import cvzone
import easyocr

# Initialize YOLO and EasyOCR
model = YOLO("best.pt")  
reader = easyocr.Reader(['en'], gpu=False)

# Initialize video capture
cap = cv2.VideoCapture("carLicence4.mp4")  


seen_plates = []

# IOU function to compare boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            
            matched = False
            for plate in seen_plates:
                if iou((x1, y1, x2, y2), plate["box"]) > 0.6:
                    matched = True
                    text = plate["text"]
                    break

            if not matched:
                cropped = frame[y1:y2, x1:x2]
                try:
                    ocr_results = reader.readtext(cropped)
                    text = ""
                    for (bbox, txt, conf) in ocr_results:
                        if conf > 0.5:
                            text = txt
                            break
                    seen_plates.append({"box": (x1, y1, x2, y2), "text": text})
                except Exception as e:
                    text = "OCR Error"
                    print(e)

            
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=3)

            
            if text:
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

           
            if box.conf is not None:
                confidence = float(box.conf[0])
                cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
