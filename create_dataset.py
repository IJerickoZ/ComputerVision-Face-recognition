#creating database
import cv2, sys, numpy, os
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  #All dataset will be present this folder

def main():
    sub_folder = input('Enter your name: ') #input your dataset name 
    openCam(sub_folder) #call openCam func

def openCam(sub_folder):
    path = os.path.join(datasets, sub_folder)
    if not os.path.isdir(path): #create new folder if folder doesn't have this name 
        os.mkdir(path)
    (width, height) = (130, 100) #pic size
    face_cascade = cv2.CascadeClassifier(haar_file) #import detect face model
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    cv2.destroyAllWindows()
    count = 1

    while count < 50: #create dataset when can detected face
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 4) #detect face. And return tuple
        for (x,y,w,h) in faces: #x = horizontal position face, y = Vertical position face, w = face length, h = face height
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2) #create rectangle on pic
            face = gray[y:y + h, x:x + w] #crop pic in rectangle
            face_resize = cv2.resize(face, (width, height)) #resize
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
    
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(150)
        if key == 27:
            cv2.destroyAllWindows()
            break
main()
