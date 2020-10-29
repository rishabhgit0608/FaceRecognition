import cv2

cam=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cam.read()
    if not ret:
        continue  
    faces=classifier.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face  # tuple unpacking
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # start ,diagnol point,color,thickness
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()