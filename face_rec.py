import cv2, sys, numpy, os
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Wait a few second...')
def main():
    (images, labels, names, index) = ([], [], {}, 0) #create new tuple
    for (subdirs, dirs, files) in os.walk(datasets): #open datasets folder
        for subdir in dirs: #open sub folder
            names[index] = subdir #set name of face when can detect
            subjectpath = os.path.join(datasets, subdir) #set all path
            for i in os.listdir(subjectpath):
                path = subjectpath + '/' + i #set pic path
                label = index #set unique num for pic
                images.append(cv2.imread(path, 0)) #add pic to array
                labels.append(int(label)) #add number to array
            index += 1
    (width, height) = (130, 100) 

    (images, labels) = [numpy.array(lis) for lis in [images, labels]] #Convert images and labels list to np array prepair train dataset
    model = cv2.face.LBPHFaceRecognizer_create() #create recognition model
    model.train(images, labels) #train datasets

    face_cascade = cv2.CascadeClassifier(haar_file) #import detect face model
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #open cam

    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #detect face. And return tuple

        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2) #create rectangle on pic
            face = gray[y:y + h, x:x + w] #crop pic in rectangle
            face_resize = cv2.resize(face, (width, height)) #resize
            prediction = model.predict(face_resize) #Compare with data in datasets 
            #prediction is an object (0 is index of subdir and 1 is predicted confidence)

            if prediction[1] > 60: #set threshold 
                #if prediction[1] > threshold will create blue rectangle and show prediction name
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3) #create blue rectangle
                cv2.putText(im,'%s' % (names[prediction[0]]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
                print("%s - %.2f"%(names[prediction[0]],prediction[1]))
            else:
                #if prediction[1] < threshold will create red rectangle and show unknown
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3) #create red rectangle
                cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
        
        cv2.imshow('Face', im)
        key = cv2.waitKey(10) #close when press esc
        if key == 27:
            cv2.destroyAllWindows()
            break
main()