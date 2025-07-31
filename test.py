import cv2
from ultralytics import YOLO
import cvzone
import easyocr


model = YOLO("best.pt")  


reader = easyocr.Reader(['en'], gpu=False)


cap = cv2.VideoCapture("carLicence4.mp4")  

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=3)
            cropped = frame[y1:y2, x1:x2]

            try:
                ocr_results = reader.readtext(cropped)
                for (bbox, text, conf) in ocr_results:
                    if conf > 0.5:
                        
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (255, 0, 255), 2)
            except Exception as e:
                print("OCR Error:", e)

            if box.conf is not None:
                confidence = float(box.conf[0])
                cv2.putText(frame, f'Conf: {confidence:.2f}', (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
