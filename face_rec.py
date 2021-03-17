import cv2, sys, numpy, os
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Wait a few second...')
def main():
    (images, labels, names, index) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[index] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for i in os.listdir(subjectpath):
                path = subjectpath + '/' + i
                label = index
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            index += 1
    (width, height) = (130, 100)

    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    model = cv2.face.LBPHFaceRecognizer_create() 
    model.train(images, labels)

    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3)

            if prediction[1]<100:
                cv2.putText(im,'%s' % (names[prediction[0]]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
                print("%s - %.2f"%(names[prediction[0]],prediction[1]))
            else:
                cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
        
        cv2.imshow('Face Recognition', im)
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            break
main()