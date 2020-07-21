# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:56:15 2020

@author: Quang
"""

import argparse
import os
import time
import cv2
import numpy as np

import dlib
from utils import CentroidTracker, TrackableObject, Conf

from bounding_box import bounding_box as bb


   
    
# augument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v" ,"--video", type=str, required=True, 
                help="Path to input video")
ap.add_argument("-c","--config", default="./config.json",
               help="Path to the input configuration file")
ap.add_argument("-s","--save",default=False,
                help="Save processed video (True/False)")
args = vars(ap.parse_args())


# initialize the lit of class labels MobileNet SSD
classes = ["di_bo","xe_dap","xe_may","xe_hang_rong","xe_ba_gac","xe_taxi","xe_hoi","xe_ban_tai","xe_cuu_thuong","xe_khach","xe_buyt","xe_tai","xe_container","xe_cuu_hoa"]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# load the configuration file
conf = Conf(args["config"])

# load our serialized model from disk
def load_model():
    print("[INFO] Loading model...")
    
    if os.path.isfile(conf["prototxt_path"]) and os.path.isfile(conf["model_path"]): 
#        net = cv2.dnn.readNet("yolov3-5c-5000-max-steps.cfg", "yolov3-5c-5000-max-steps_final.weights")
        #change the path to config, and model in config file  
        net = cv2.dnn.readNet(conf["prototxt_path"],conf["model_path"])

        print("[INFO] Loaded model successfully...")
        return net
    else:
        print("Model is not found...")
            
# main function 
def main():
    
    # initialize a list storing counted objectID
    counted_objectID = []
    
    # intialize frame dimensions
    height,width = None, None
    
    """
     initialize our centroid tracker, then initialize a lost to store
     each of our dlib correlation trackers, followed by a dictionary to 
     map each unique object ID to a TrackableOject 
    """ 
    ct = CentroidTracker(maxDisappeared=conf["max_disappear"],
                         maxDistance=conf["max_distance"])
    trackers = []
    trackableObjects = {}
    
    # keep the count of total number of frames
    totalFrame = 0
    
    net = load_model()
    cap = cv2.VideoCapture(args["video"])
    time.sleep(1)
    
    # initilize the coordiate of counting line
    _, frame = cap.read()
    x_low = 400
    y_low = frame.shape[0]-conf["line_coordinate"]
    x_high = 1250
    y_high = y_low - 700
    rec_line = [(x_low, y_low ),(x_high,y_high)]
                  
    
    # initialize car counting variable
    car_count = 0
    
    if args["save"]:
        video_size = (frame.shape[1]+250,frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter("processed_video.avi",fourcc,24,video_size)

    while True:
        ret, frame = cap.read()
        # coordinate of countign line
        
        # brekout the loop if no frame is captured
        if frame is None:
            break
       
        # save origin frame for later displaying
        origin_frame = frame.copy()
        
        # crop frame before countign line
        frame = frame[y_low - 700:y_low, x_low : x_high]
        
        # convert to RGB colour space
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        # if the frame dimensions are empty, set them
        if width is None or height is None:
            (height,width) = frame.shape[:2]
        """
        initialize our list of bounding box rectangles returned by 
        either (1) our object detector or (2) the correlation trackers
        """
        rects = []
        
        """
        check to see if we should run a more computationally expensive
        object detection method to add our tracker
        """
        if totalFrame % conf["track_object"] == 0:
            # initialize our new set of object trackers
            trackers = []
            
            """
            convert the frame to a blob and pass the blob 
            through the netwrok and obtain the detections
            """
            #blob = cv2.dnn.blobFromImage(frame, size=(300,300),ddepth=cv2.CV_8U)
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            #net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,127.5,127.5])
            net.setInput(blob)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)
            # loop over the detections
            boxes = []
            confidences = []
            class_ids = []
            for out in outs:
                for detection in out:
                    """
                    extract the confidence (i.e., probability)
                    associate with the predicton
                    """
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    """
                    filter out weak detections by 
                    setting a threshold confidence
                    """
                    if confidence > conf['confidence']:
                        """
                        extract the index of the class label
                        from detection list
                        """
                        # if the class label is not a car, skip it
                        """
                        compute the (x,y)-coordinates of the 
                        bounding box for the object
                        """
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        #box = detections[i,:4] * np.array([W,H,W,H])
                        boxes.append([x, y, w, h])
                        confidences.append (float (confidence))
                        class_ids.append (class_id)
                        (startX, startY, endX, endY) = (x, y, x+w, y+h)
                        
                        
                        """
                        construct a dlib rectangle object from the bounding
                        box coordinates and then start the dlib correlation tracker
                        """
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX,startY, endX, endY)
                        tracker.start_track(rgb,rect)
                        
                        """
                        add the tracker to our list of trackers
                        so we can utilize it during skip frames
                        """
                        trackers.append(tracker)
                """
            otherwise, we should utilize our object "trackes" rather than 
            object "detectors" to obtain a higher frame preprocessing
                """
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        else:
            # loop over the tracker
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()
                
                # unpack the position project
                post_list = [pos.left(),pos.top(),pos.right(), pos.bottom()]
                [startX, startY, endX, endY] = list(map(int,post_list))         
                
                # add the bounding box coordinate to the rectangle list
                rects.append((startX,startY,endX,endY))
                
        """
        use the centroid tracker to associate the (1) old object
        centroids with (2) the newly computed object centroids
        """
        objects = ct.update(rects)
        
        # loop over the tracked objects
        for (objectID, centroid),rect in zip(objects.copy().items(),rects):
            
            # if objectID is already counted then skip it
            if objectID in counted_objectID:
                ct.deregister(objectID)
                rects.remove(rect)
                trackers.remove(tracker)
                objects = ct.update(rects)
                break
            else:
                """
                if centroid of the car cross count line 
                then increment car_count
                """
                if (centroid[1] + 60 > rec_line[0][1]): 
                    rects.remove(rect)
                    trackers.remove(tracker)
                    counted_objectID.append(objectID)
                    ct.deregister(objectID)
                    objects = ct.update(rects)
                    car_count+=1
                    break
            
            """
            check to see if a trackable object exists
            for the current object ID
            """
            to = trackableObjects.get(objectID, None)
            
            # if there is no exisiting trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
            
            # store the trackable object in our dictinonary
            trackableObjects[objectID] = to
            
            
            
            """
            draw both the ID of the object and the centroid 
            of the object on the output frame
            """
            text = "ID {}".format(objectID)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str (classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(origin_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(origin_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(origin_frame, text, (centroid[0] - 10, centroid[1] - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(origin_frame, (centroid[0], centroid[1]), 4,
                       (0, 255, 0), -1)
            
        
        # create a blank space next to frame to display No. cars
        blank_region = np.ones((origin_frame.shape[0],250,3), np.uint8)*255
        cv2.putText(blank_region, "No. car(s):", (40,origin_frame.shape[0]//2-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
    
        cv2.rectangle(origin_frame,rec_line[0],rec_line[1],(0,255,0),3)
        
        # stack the frame with blank space
        cv2.putText(blank_region, str(car_count), (40,origin_frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 3)
        
        stack_image = np.concatenate((origin_frame,blank_region),axis=1)
        cv2.imshow("Final result", stack_image)
        
        # save processed videos
        if args["save"]:
            writer.write(stack_image)
            
        # press ESC to terminate the system
        key = cv2.waitKey(1)  & 0xff
        
        if key == 27:
            break
        
        # increment the total number of frame processed so far 
        totalFrame += 1
        
        
    cap.release()
    cv2.destroyAllWindows()
    if args["save"]:
        writer.release()
           
if __name__=="__main__":
    main()
