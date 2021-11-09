import cv2

cap = cv2.VideoCapture("body.mp4")

body_detection = cv2.CascadeClassifier("fullbody.xml")

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    bodies = body_detection.detectMultiScale(gray,1.1,1)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Body Capture",frame)
    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()