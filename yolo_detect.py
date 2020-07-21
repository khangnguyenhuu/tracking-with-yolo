import cv2
import numpy as np
import os
import random
import imutils

net = cv2.dnn.readNet("yolov3-5c-5000-max-steps_7000.weights", "yolov3-5c-5000-max-steps.cfg")
output_path = os.path.join("results", "my_result.jpg")
# Name custom object
classes = ["di_bo","xe_dap","xe_may","xe_hang_rong","xe_ba_gac","xe_taxi","xe_hoi","xe_ban_tai","xe_cuu_thuong","xe_khach","xe_buyt","xe_tai","xe_container","xe_cuu_hoa"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def detect_potHole(img):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # output_image = img[y:(y+h), x:(x+w), :]
    
    # Store image
    cv2.imwrite(output_path, img)   

    return img, output_path

def detect_video(video_path):
    # start detect video
    cap = cv2.VideoCapture(video_path)
    while (True):
        ret, frame = cap.read()
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(origin_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(origin_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)    
        cv2.imshow('car_', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists("results"):
        os.mkdir("results")

    detect_video('IMG_3124.mp4')
    # image, image_path = detect_potHole(cv2.imread("images/img-3.jpg"))
    # cv2.imshow("demo", image)
    # cv2.waitKey(0)