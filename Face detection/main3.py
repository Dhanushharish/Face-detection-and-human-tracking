import cv2
import numpy as np

# Load the pre-trained YOLOv3 model and configuration file
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Load COCO class names (for object class labeling)
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Set the model's input size
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize the video capture
vid = input("Enter video Name :")
if vid=='cam':
    vid=0
cap = cv2.VideoCapture(vid)  # Replace 'video.mp4' with your video source or use 0 for a live camera feed

prev_boxes = []  # Store the previous frame's detected person boxes

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setInput(blob)
    
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)
    
    class_ids = []
    confidences = []
    boxes = []
    
    # Process the model's output to get detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # Apply non-maximum suppression to filter out overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            if label == 'person':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            human_roi = frame[y:y + h, x:x + w]
            if not human_roi.size == 0:  # Check if human_roi is not empty
                gray = cv2.cvtColor(human_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(human_roi, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
                    roi_gray = gray[fy:fy + fh, fx:fx + fw]
                    roi_color = human_roi[fy:fy + fh, fx:fx + fw]

                    # Detecting eyes in the detected faces
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)
    
    prev_boxes = boxes  # Update the previous frame's detected person boxes
    
    # Display the output frame
    cv2.imshow('Human Tracking', frame)
    
    # Exit the program with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
