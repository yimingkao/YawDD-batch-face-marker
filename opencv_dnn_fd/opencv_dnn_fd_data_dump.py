
import cv2
import pandas as pd
import numpy as np
#import time
#import imutils

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
#video_path = '../../YawDD/Mirror/Male/'
#csv_file = '../../YawDD/Mirror/Male_all.csv'
video_path = '../../YawDD/Mirror/Female/'
csv_file = '../../YawDD/Mirror/Female_all.csv'

data = pd.read_csv(csv_file)
for i in range(len(data)):
    filename = video_path + data['Name'][i]
    dstname = data['Name'][i].replace('avi', 'csv')
    vin = cv2.VideoCapture(filename)
    length = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    print('{}: {}'.format(filename, length))
    csvf = open(dstname, 'w')
    csvout = 'frame,confidence,sx,sy,ex,ey\n'
    csvf.write(csvout)
    for j in range(length):
        ret, frame = vin.read()
    
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
     
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        #print(detections.shape)
        #print(detections[0,0,0,:])
        # loop over the detections
        for k in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, k, 2]
    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue
    
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, k, 3:7] * np.array([w, h, w, h])
            csvout = '%d,%f,%f,%f,%f,%f\n'%(j, confidence, box[0], box[1], box[2], box[3])
            csvf.write(csvout)
#            (startX, startY, endX, endY) = box.astype("int")
     
            # draw the bounding box of the face along with the associated
            # probability#            text = "{:.2f}%".format(confidence * 100)
#            y = startY - 10 if startY - 10 > 10 else startY + 10
#            cv2.rectangle(frame, (startX, startY), (endX, endY),
#                (0, 0, 255), 2)
#            cv2.putText(frame, text, (startX, y),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            break #one face only
    vin.release()
    #break