import cv2

img = cv2.imread("body.jpg")
body_detection = cv2.CascadeClassifier("fullbody.xml")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

body = body_detection.detectMultiScale(gray,1.2,1)

print(body)
for (x,y,w,h) in body:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("Bodey detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()