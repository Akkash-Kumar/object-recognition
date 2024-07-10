import numpy    #array calculations

import imutils  #resize image

import cv2   #opencv library


protoTxt = 'MobileNetSSD_deploy.prototxt.txt'     #initialise file

model = 'MobileNetSSD_deploy.caffemodel'    #initialise model file

confidentThres = 0.2    #initialise confident threshold to find whether it is object

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobile"]                                  #labels of output which model have trained

COLORS = numpy.random.uniform(0,255,size = (len(CLASSES),3))    #to draw box and put text in multiple colors when multiple objects detected in one frame
                                                                 #3 - rgb, 0 to 255 - pick any color randomly

net = cv2.dnn.readNetFromCaffe(protoTxt,model)   #load caffe model

camera = cv2.VideoCapture(0)   #primary camera initialise

while True :    #to run camera continuously

    _,frame = camera.read()  # to read frame from camera

    frame = imutils.resize(frame,width = 500)   #to resize camera frame

    (h,w) = frame.shape[:2]    #to get height and width of frame

    #print("height and width",h,w)

    imResizeBlob = cv2.resize(frame,(300,300))   #resize image inside camera frame

    blob = cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)   #to convert to blob image

    net.setInput(blob)   #to set blob image as input to caffe model

    detections = net.forward()  #call forward function to get id,confident level,coordinates

    detShape = detections.shape[2]

    #print("detections",detections)

    for i in numpy.arange(0,detShape):

        confidence = detections[0,0,i,2]   # 2- get confidence level

        if confidence > confidentThres:

            idx = int(detections[0,0,i,1])   #1 - to get id

            print("Class id:", detections[0,0,i,1])   #print id

            box = detections[0,0,i,3:7] * numpy.array([w,h,w,h])   #3:7 to get coordinates

            (startX,startY,endX,endY) = box.astype('int')   #change to int values

            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)    #get names of objects in classes with confidence

            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)   #to draw rectangle

            if startY -15 > 15:    #when object zoom in , text will appear inside box

                y= startY - 15

            else :

                startY + 15

            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)   #to display text above  box

    cv2.imshow("frame",frame)   #display frame

    key = cv2.waitKey(10)    #wait for 10 frames

    if key == 27:   #esc key is pressed,close camera
        break


camera.release()   #release camera

cv2.destroyAllWindows()  #close window

            

        

    

    






