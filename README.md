# YawDD Batch Face Marker
The [YawDD](http://www.site.uottawa.ca/~shervin/yawning/) is a dataset that contains video of yawning people. I used this dataset as the final project of the AI school that I want to analysis the fatigue degree. For getting more information to train the NN model, I tried to use different utitlities to extract the information from video automatically.
The dataset is assumed to be placed as the same folder of this project.

## Installation of opencv
I used the anaconda environment so the following command did all.

  $ conda install -c https://conda.anaconda.org/opencv opencv
  
## Installation of dlib
The dlib could be installed by the same way exception the different channel

  $ conda install -c conda-forge dlib
  
Some library will be downgraded or removed to resolve the conflict.

## opencv-dlib-fd
This marker code uses the traditional [opencv Haar cascades](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html) detection to get the face and eye bounding boxes. This method requires the face model and eye model which could be downloaded from https://github.com/opencv/opencv/tree/master/data/haarcascades. Just download the haarcascade_frontalface_default.xml and haarcascade_eye.xml.
The better solution is [dlib](http://dlib.net/) which is implemented by C++ originally. It can detect the 68 face landmarks and give the bounding box of face also. You need to download the extra two model files:
- [mmod_human_face_detector.dat](https://github.com/davisking/dlib-models/blob/master/mmod_human_face_detector.dat.bz2)
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
I also referenced the resource on internet to get the yaw, roll, pitch information.

## opencv-dnn-fd
Opencv also have modren DNN models for face detection. It provides much better performance (compared to the haar method). They used the Single Shot Detection (SSD) as the model and trained by caffe framework. You need to download the following two files to use it:
- [deploy.prototxt.txt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
- [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

## openface-landmark
The [openface](https://github.com/TadasBaltrusaitis/OpenFace/wiki) is a toolkit for checking the facial behavior. It used the MTCNN model to detect the face. The download link is https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_v2.0.3/OpenFace_2.0.3_win_x86.zip. You also need to get 4 more files for execution successfully.
- [cen_patches_0.25_of.dat](https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1)
- [cen_patches_0.35_of.dat](https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1)
- [cen_patches_0.50_of.dat](https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1)
- [cen_patches_1.00_of.dat](https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1)
The downloaded file should be put the following path
  
  model\patch_experts


- 
