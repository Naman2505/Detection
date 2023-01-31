import cv2  #importing OpenCV

alg = "haarcascade_frontalface_default.xml" #Accessing pre-trained model
haar_cascade = cv2.CascadeClassifier(alg) #loading the model with cv2

cam = cv2.VideoCapture(0) #initializing camera

while True:
    _,img = cam.read() #reading the frame from the camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting color into gray scale

    face = haar_cascade.detectMultiScale(grayImg,1.3,4) #get coordinate of face

    for(x,y,w,h) in face: #segregating x,y,w,h
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),2)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == 27:    #it will break if and only i press escape key
        break
cam.release()
cv2.destroyAllWindows()
