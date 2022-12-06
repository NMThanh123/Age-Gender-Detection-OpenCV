#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
#           Họ và tên                   MSSV                                    # 
#        Nguyễn Minh Thành            20146422                                  #
#                                                                               #
#        Nguyễn Hông Nhân             20146381                                  #
#                                                                               #
#          Trảo An Tân                20146416                                  #
#                                                                               #
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''#
import cv2 as cv
from keras.models import load_model
import numpy as np
 


# Loading the required haar-cascade xml classifier file
haar_cascade = cv.CascadeClassifier('face.xml')

# Load model to detect gender and age
model = load_model('age_and_gender.h5')

# faces_rect = haar_cascade.detectMultiScale(img, 1.15, 9)

# main
def detect():
    cap = cv.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.resize(frame,(900, 500))
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.15, 9)

        # draw rectangle to surround face
        for (x, y, w, h) in faces_rect:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.rectangle(frame, (x, y-2), (x+w, y-20), (255, 255, 255), -1)
            face_color = frame[y:y+h, x:x+w]
            face_color = cv.resize(face_color,(128, 128))

            # Converting image_face to grayscale and nomalize 
            gray = cv.cvtColor(face_color, cv.COLOR_BGR2GRAY)
            gray = np.array(gray)
            gray = gray/255.0

            # predict 
            age_gender = model.predict(gray.reshape(1, 128, 128, 1))
            # round result for gender and age
            gender = round(age_gender[0][0][0])
            age = round(age_gender[1][0][0])

            gender_label_position = (x,y-3)
            age_label_position = (x,y+h+20)
            # draw text
            if   0 < age < 15:
                cv.putText(frame, "Gender:{}  ".format(gender), gender_label_position ,cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # cv.putText(frame, "Age={}  ".format(age), age_label_position, cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv.imshow('Detected faces', frame) 
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    detect()
