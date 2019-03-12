
# Reference
# http://nbviewer.jupyter.org/github/erhwenkuo/deep-learning-with-keras-notebooks/blob/master/7.6-head-pose-estimation.ipynb

import numpy as np
import cv2
import dlib
import pandas as pd
#from imutils import face_utils
#import imutils

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
    
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    return coords

def landmarks_to_facepoints(landmarks):
    noise_tip = landmarks[33:34]
    chin = landmarks[8:9]
    left_eye_corner = landmarks[36:37]
    right_eye_corner = landmarks[45:46]
    left_mouth_corner = landmarks[48:49]
    right_mouth_corner = landmarks[54:55]
    face_points = np.concatenate((noise_tip, chin,
                                  left_eye_corner, right_eye_corner,
                                  left_mouth_corner, right_mouth_corner))
    face_points = face_points.astype(np.double)
    return face_points

def pose_estimate(face_points, img):
    #Generic 3D model
    model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
            ])
    #Camera parameters
    focal_length = img.shape[1]
    center = (img.shape[1]/2, img.shape[0]/2)
    camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             #[0, center[1]/2, center[1]],
             [0, 0, 1]], dtype = "double")
    dist_coeffs = np.zeros((4,1))
    #OpenCV to calculate rotation & shift
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, face_points, camera_matrix,
            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #Euler angles
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
    yaw   = eulerAngles[1]
    pitch = eulerAngles[0]
    roll  = eulerAngles[2]
    if pitch > 0:
      pitch = 180 - pitch
    elif pitch < 0:
      pitch = -180 - pitch
    yaw = -yaw
    
    sx = 20
    sy = img.shape[0] - 60
    text = "up(+)/dn(-) [pitch]: %.2f" % (pitch)               
    cv2.putText(img, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 1, cv2.LINE_AA)
    sy += 20
    text = "ry(+)/ly(-) [yaw]  : %.2f" % (yaw)               
    cv2.putText(img, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 1, cv2.LINE_AA)
    sy += 20
    text = "rr(+)/lr(-) [roll] : %.2f" % (roll)               
    cv2.putText(img, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 1, cv2.LINE_AA)    
                    
    # 投射一個3D的點 (100.0, 0, 0)到2D圖像的座標上
    (x_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(100.0, 0.0, 0.0)]), rotation_vector,
            translation_vector, camera_matrix, dist_coeffs)
    # 投射一個3D的點 (0, 100.0, 0)到2D圖像的座標上
    (y_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 100.0, 0.0)]), rotation_vector,
            translation_vector, camera_matrix, dist_coeffs)
    # 投射一個3D的點 (0, 0, 100.0)到2D圖像的座標上
    (z_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 100.0)]), rotation_vector,
            translation_vector, camera_matrix, dist_coeffs)

    # 以 Nose tip為中心點畫出x, y, z的軸線
    p_nose = (int(face_points[0][0]), int(face_points[0][1]))
    p_x = (int(x_end_point2D[0][0][0]), int(x_end_point2D[0][0][1]))
    p_y = (int(y_end_point2D[0][0][0]), int(y_end_point2D[0][0][1]))
    p_z = (int(z_end_point2D[0][0][0]), int(z_end_point2D[0][0][1]))
    cv2.line(img, p_nose, p_x, (0,0,255), 3)  # X軸 (紅色)
    cv2.line(img, p_nose, p_y, (0,255,0), 3)  # Y軸 (綠色)
    cv2.line(img, p_nose, p_z, (255,0,0), 3)  # Z軸 (藍色)
    # 把6個基準點標註出來
    for p in face_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (255,255,255), -1)    

    return pitch[0], yaw[0], roll[0]

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

vin = cv2.VideoCapture(0)

while True:
    ret, frame = vin.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects, scores, idx = detector.run(frame, 0)
    # Dlib results
    for k, d in enumerate(face_rects):
        shape = predictor(gray, d)
        shape = shape_to_np(shape)
        x, y, w, h = rect_to_bb(d)
        text = "%2.2f(%d)" % (scores[k], idx[k])                    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x+w, y+h), cv2.FONT_HERSHEY_DUPLEX,
            0.7, (255, 255, 255), 1, cv2.LINE_AA)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        if len(shape) == 68:
            face_points = landmarks_to_facepoints(shape)
            pitch, yaw, roll = pose_estimate(face_points, frame)
            mouth = shape[48:]
            ml = np.amin(mouth, axis=0)
            mr = np.amax(mouth, axis=0)
            cv2.rectangle(frame,(ml[0]-2,ml[1]-2),(mr[0]+2,mr[1]+2),(0,255,0),2)
            inner = np.concatenate((shape[61:64], shape[65:68]))
            iminy = np.amin(inner, axis=0)[1]
            imaxy = np.amax(inner, axis=0)[1]
            cv2.line(frame, (shape[62][0], iminy), (shape[62][0], imaxy), (0,255,0), 2)
        else:
            print('Frame %d has %d landmarks only'%(j, len(shape)))
        break # only get one face result
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vin.release()
cv2.destroyAllWindows()