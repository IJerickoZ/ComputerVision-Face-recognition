#creating database
import cv2, numpy, os
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

def main():
    subdir = input('Enter your name: ')
    openCam(subdir) #เรียกใช้ openCam

def openCam(subdir):
    path = os.path.join(datasets, subdir)
    if not os.path.isdir(path): #สร้างโฟลเดอร์ใหม่ถ้าไม่มี Path
        os.mkdir(path)
    (width, height) = (130, 100) #กำหนดขนาด
    face_cascade = cv2.CascadeClassifier(haar_file) 
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #เรียกใช้ webcam
    cv2.destroyAllWindows()
    count = 1

    while count < 50:
        (_, im) = webcam.read() #อ่านข้อมูลจาก webcam
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #แปลงเป็นภาพขาวดำ
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) #หาใบหน้า และคืนค่าทูเพิ้ลออกมา
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2) #สร้างสี่เหลี่ยมบนใบหน้า
            face = gray[y:y + h, x:x + w] #ครอบพื้นที่ในสี่เหลี่ยม
            face_resize = cv2.resize(face, (width, height)) #resize
            cv2.imwrite('%s/%s.png' % (path,count), face_resize) #save รูปลงโฟลเดอร์ย่อย
        count += 1
    
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(150)
        if key == 27:
            cv2.destroyAllWindows()
            break
main()
