import cv2
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
from datetime import datetime

model = YOLO("best.pt")
names = model.names
ocr = PaddleOCR(use_angle_cls=True, lang='en')

now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"plates_{now_str}.txt"

saved_ids = set()
id_to_plate = {}    

def RGB(event,x,y,flag,param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to ({x}, {y})")
    
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", RGB)

cap = cv2.VideoCapture("carLicence1.mp4")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1000, 600))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            c = names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(frame,f'{c.upper()}', (x1, y1 - 10), scale=1, thickness=2, offset=5, border=2, colorR=(0, 0, 255), colorT=(255, 255, 255))

            if c.lower() == "licence":
                cropped_plate = frame[y1:y2, x1:x2]
                if cropped_plate.size <= 0:
                    continue

                if track_id not in id_to_plate:
                    results = ocr.predict(cropped_plate)
                    plane_text = ""
                    for line in results:
                        if line in results:
                            if line:
                                plane_text += ''.join([word_info[1][0] for word_info in line]) + " "
                    plane_text = plane_text.strip()

                    if plane_text:
                        id_to_plate[track_id] = plane_text
                        if track_id not in saved_ids:
                            with open(log_filename, "a") as f:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                f.write(f"{timestamp} | ID: {track_id}, Plate: {plane_text}\n")
                            saved_ids.add(track_id)
                            print(f"Saved plate for ID {track_id}: {plane_text}")
                
                if track_id in id_to_plate:
                    plate_text = id_to_plate[track_id]
                    label = f"ID: {track_id}| Plate: {plate_text}"
                    cvzone.putTextRect(frame, label, (x1, y2 + 10), scale=1, thickness=2, offset=5, border=2, colorR=(0, 255, 0), colorT=(255, 255, 255))
                else:
                    cvzone.putTextRect(frame, f"ID: {track_id}| Plate: Unknown", (x1, y2 + 10), scale=1, thickness=2, offset=5, border=2, colorR=(0, 255, 0), colorT=(255, 255, 255))
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()         