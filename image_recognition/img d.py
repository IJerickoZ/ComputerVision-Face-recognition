import cv2, numpy, os
haar_file = 'haarcascade_frontalface_alt2.xml'
(width, height) = (130, 100) 
datasets = 'datasets'
print('Wait a few second...')
def main():
    (images, labels, names, index) = ([], [], {}, 0) #สร้างทูเพิ้ลเปล่า
    for (subdirs, dirs, files) in os.walk(datasets): #ลูปเข้าโฟลเดอร์ดาต้าเซ็ตเพื่อเตรียมอ่านข้อมูล
        for subdir in dirs: #ลูปเพื่อเตรียมเข้าไปอ่านข้อมูลในโฟลเดอร์ย่อย
            names[index] = subdir #ตั้งชื่อของกลุ่มข้อมูลให้มีชื่อเดียวกับโฟลเดอร์ย่อย
            subjectpath = os.path.join(datasets, subdir) #Save Path ที่อยู่ในปัจจุบัน
            for i in os.listdir(subjectpath): #เข้ามาในโฟลเดอร์ย่อย
                path = subjectpath + '/' + i #Save picture path
                label = index #กำหนด index ให้กับรูปภาพในแต่ละโฟลเดอร์ย่อย
                images.append(cv2.imread(path, 0)) #เพิ่มภาพเข้า array แบบขาวดำ
                labels.append(int(label)) #เพิ่มเข้า array
            index += 1

    (images, labels) = [numpy.array(lis) for lis in [images, labels]] #รวม images กับ labels ให้กลายเป็น array เดียว
    model = cv2.face.LBPHFaceRecognizer_create() #สร้าง LBPH recognition model
    model.train(images, labels) #train model ด้วยชุด array 

    face_cascade = cv2.CascadeClassifier(haar_file) #import haar cascade

    while True:
        # im = cv2.imread('C:\\Users\\Piya Kaewket\\Desktop\\MiniProjectComVision\\image_recognition\\input\\a.jpg')
        # im = cv2.imread('C:\\Users\\Piya Kaewket\\Desktop\\MiniProjectComVision\\image_recognition\\input\\b.jpg')
        # im = cv2.imread('C:\\Users\\Piya Kaewket\\Desktop\\MiniProjectComVision\\image_recognition\\input\\d.jpg')
        im = cv2.imread('C:\\Users\\Piya Kaewket\\Desktop\\MiniProjectComVision\\image_recognition\\input\\d.jpg')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #แปลงภาพเป็นขาวดำ

        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #ค้นหาใบหน้าด้วย model ที่ import เข้ามา โดยจะ return ค่าออกมาเป็นทูเพิ้ล

        for (x,y,w,h) in faces: #ลูปข้อมูลในทูเพิ้ล
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2) #สร้างสี่เหลี่ยมในภาพด้วยข้อมูลจากทูเพิ้ล
            face = gray[y:y + h, x:x + w] #ครอบภาพในพื้นที่สี่เหลี่ยม
            face_resize = cv2.resize(face, (width, height))
            prediction = model.predict(face_resize) #ทำการเปรียบเทียบกับตัวข้อมูลในโมเดล
            #prediction จะได้กลับมาเป็นค่า Array (0 คือชื่อของ subdir และ 1 คือค่าความมั่นใจ(ยิ่งน้อยยิ่งดี))

            if prediction[1] < 85: #set threshold 
                #ถ้า prediction[1] < threshold จะสร้างสี่เหลี่ยมสีน้ำเงินล้อมรอบใบหน้า และแสดงชื่อที่ทำนายไว้
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3) #create blue rectangle
                cv2.putText(im,'%s' % (names[prediction[0]]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
                print("%s - %.2f"%(names[prediction[0]],prediction[1]))
            else:
                #ถ้า prediction[1] > threshold จะสร้างสี่เหลี่ยมสีแดง และแสดงชื่อเป็น unknown
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 3) #create red rectangle
                cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
        
        cv2.imshow('Face', im)
        key = cv2.waitKey(10)
        if key == 27:
            cv2.destroyAllWindows()
            break
main()